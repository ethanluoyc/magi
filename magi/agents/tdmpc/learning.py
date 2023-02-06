import dataclasses
import functools
import time
from typing import Iterator, NamedTuple, Optional, Tuple

import acme
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb
import tree
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.jax import utils as jax_utils
from acme.utils import async_utils
from acme.utils import counting
from acme.utils import loggers

from magi.agents.tdmpc import networks as tdmpc_networks

TDMPCReplaySample = jax_utils.PrefetchingSplit


@dataclasses.dataclass(frozen=True)
class LossScalesConfig:
    consistency: float = 2.0
    reward: float = 0.5
    value: float = 0.1


class TrainingState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState
    key: jax.random.PRNGKeyArray
    steps: int


class TDMPCLearner(acme.Learner):
    def __init__(
        self,
        spec: specs.EnvironmentSpec,
        networks: tdmpc_networks.TDMPCNetworks,
        random_key: jax.random.PRNGKeyArray,
        replay_client: reverb.Client,
        iterator: Iterator[TDMPCReplaySample],
        *,
        optimizer: optax.GradientTransformation,
        discount: float = 0.99,
        min_std: float = 0.05,
        per_beta: float = 0.4,
        tau: float = 0.01,
        loss_scale: Optional[LossScalesConfig] = None,
        rho: float = 0.5,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):
        if loss_scale is None:
            loss_scale = LossScalesConfig()
        self._discount = discount
        self._min_std = min_std
        self._tau = tau
        self._loss_scale = loss_scale
        self._rho = rho
        self._networks = networks
        self._optimizer = optimizer
        self._per_beta = per_beta
        self._replay_client = replay_client

        param_key, key = jax.random.split(random_key)
        params = tdmpc_networks.init_params(self._networks, spec, param_key)
        opt_state = self._optimizer.init(params)
        self._state = TrainingState(
            params=params,
            target_params=params,
            opt_state=opt_state,
            key=key,
            steps=0,
        )

        self._counter = counter or counting.Counter()
        self._logger = logger

        def update_priorities(keys_and_priorities: Tuple[jnp.ndarray, jnp.ndarray]):
            keys, priorities = keys_and_priorities
            keys, priorities = tree.map_structure(
                # Fetch array and combine device and batch dimensions.
                lambda x: jax_utils.fetch_devicearray(x).reshape((-1,) + x.shape[2:]),
                (keys, priorities),
            )
            replay_client.mutate_priorities(
                table=adders_reverb.DEFAULT_PRIORITY_TABLE,
                updates=dict(zip(keys, priorities)),
            )

        self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

        self._iterator = iterator

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    def save(self):
        return self._state

    def restore(self, state):
        self._state = state

    def _compute_loss(
        self,
        params: tdmpc_networks.TDMPCParams,
        target_params: tdmpc_networks.TDMPCParams,
        batch: reverb.ReplaySample,
        key: jax.random.PRNGKeyArray,
    ):
        samples: adders_reverb.Step = jax_utils.batch_to_sequence(batch.data)
        observations = samples.observation
        actions = samples.action[:-1]
        rewards = jnp.expand_dims(samples.reward[:-1], axis=-1)

        def policy(params, obs, key):
            return self._networks.pi(params, obs, self._min_std, key)

        def next_core(action, z):
            next_z, reward = self._networks.next(params, z, action)
            return (next_z, reward), next_z

        batched_policy = jax.vmap(policy, in_axes=(None, 0, 0))
        batched_critic = jax.vmap(self._networks.q, (None, 0, 0))
        batched_encoder = jax.vmap(self._networks.h, (None, 0))

        # [H+1, B, ...]
        horizon = observations.shape[0] - 1

        z_prior = batched_encoder(params, observations)
        z_target = batched_encoder(target_params, observations)

        # z_tp1, z_tp2 ... z_tpH
        (online_z_posterior, reward_pred), _ = hk.static_unroll(
            next_core, actions, z_prior[0]
        )
        # [H, B, Z]
        assert online_z_posterior.shape[0] == horizon
        # [H+1, B, Z]

        z_predictions = jnp.concatenate([z_prior[:1], online_z_posterior], axis=0)
        # [H, B, 1]
        q1_t, q2_t = batched_critic(params, z_predictions[:-1], actions)

        key, policy_key = jax.random.split(key)
        policy_a_tp1 = batched_policy(
            params, z_prior[1:], jax.random.split(policy_key, z_prior[1:].shape[0])
        )

        q1_tp1, q2_tp2 = batched_critic(target_params, z_prior[1:], policy_a_tp1)

        td_target = rewards + jax.lax.stop_gradient(
            self._discount * jnp.minimum(q1_tp1, q2_tp2)
        )

        # Compute model loss
        consistency_loss = jnp.mean(_l2_loss(z_predictions[1:], z_target[1:]), axis=-1)
        reward_loss = jnp.squeeze(_l2_loss(reward_pred, rewards), axis=-1)
        value_loss = jnp.squeeze(
            _l2_loss(q1_t, td_target) + _l2_loss(q2_t, td_target),
            axis=-1,
        )
        priorities = jnp.squeeze(
            _l1_loss(q1_t, td_target) + _l1_loss(q2_t, td_target),
            axis=-1,
        )

        rhos = jnp.reshape(
            jnp.power(self._rho, jnp.arange(observations.shape[0])), (-1, 1)
        )
        chex.assert_equal_shape([reward_loss, value_loss, priorities, consistency_loss])

        consistency_loss = jnp.sum(rhos[:-1] * consistency_loss, axis=0)
        reward_loss = jnp.sum(rhos[:-1] * reward_loss, axis=0)
        value_loss = jnp.sum(rhos[:-1] * value_loss, axis=0)
        priorities = jnp.sum(rhos[:-1] * priorities, axis=0)

        probabilities = batch.info.probability
        importance_sampling_weights = (1 / probabilities).astype(jnp.float32)
        importance_sampling_weights **= self._per_beta
        importance_sampling_weights /= jnp.max(importance_sampling_weights)

        model_loss = (
            self._loss_scale.consistency * jnp.clip(consistency_loss, 0, 1e4)
            + self._loss_scale.reward * jnp.clip(reward_loss, 0, 1e4)
            + self._loss_scale.value * jnp.clip(value_loss, 0, 1e4)
        )

        weighted_model_loss = importance_sampling_weights * model_loss
        weighted_model_loss = jnp.mean(weighted_model_loss)
        weighted_model_loss = optax.scale_gradient(weighted_model_loss, 1.0 / horizon)

        frozen_params = jax.lax.stop_gradient(params)
        z_policy = jax.lax.stop_gradient(z_predictions)
        policy_actions = batched_policy(
            params, z_policy, jax.random.split(key, z_predictions.shape[0])
        )

        # Compute policy loss
        policy_q1, policy_q2 = jax.vmap(self._networks.q, (None, 0, 0))(
            frozen_params, z_policy, policy_actions
        )
        policy_q = jnp.squeeze(jnp.minimum(policy_q1, policy_q2), axis=-1)
        policy_loss = -policy_q
        chex.assert_rank([rhos, policy_q], 2)
        chex.assert_equal_rank([rhos, policy_q])
        policy_loss = jnp.mean(jnp.sum(rhos * policy_loss, axis=0))

        total_loss = weighted_model_loss + policy_loss

        metrics = {
            "total_loss": jnp.mean(total_loss),
            "policy_loss": jnp.mean(policy_loss),
            "model/model_loss": jnp.mean(model_loss),
            "model/weighted_model_loss": jnp.mean(weighted_model_loss),
            "model/consistentcy_loss": jnp.mean(consistency_loss),
            "model/reward_loss": jnp.mean(reward_loss),
            "model/critic_loss": jnp.mean(value_loss),
        }

        return total_loss, (priorities, metrics)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update(self, state, batch):
        key, random_key = jax.random.split(state.key)
        grad_fn = jax.value_and_grad(self._compute_loss, has_aux=True)
        (_, (priorities, metrics)), gradients = grad_fn(
            state.params, state.target_params, batch, key
        )

        metrics["grad_norm"] = optax.global_norm(gradients)

        updates, opt_state = self._optimizer.update(
            gradients, state.opt_state, state.params
        )

        params = optax.apply_updates(state.params, updates)
        target_params = optax.incremental_update(params, state.target_params, self._tau)

        steps = state.steps + 1
        new_state = TrainingState(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            key=random_key,
            steps=steps,
        )

        return (new_state, metrics, priorities)

    def step(self):
        prefetching_splits = next(self._iterator)

        keys = prefetching_splits.host
        samples = prefetching_splits.device

        (self._state, metrics, priorities) = self._update(self._state, samples)

        if self._replay_client:
            self._async_priority_updater.put((keys, priorities))

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        counts = self._counter.increment(steps=1, walltime=elapsed_time)

        if self._logger is not None:
            self._logger.write({**counts, **metrics})

    def get_variables(self, names):
        variables = {"policy": self._state.params}
        return [variables[name] for name in names]


def _l2_loss(predictions, targets):
    # MSE loss, unlike the one used in optax, it does not have the 0.5 at the front.
    chex.assert_type([predictions], float)
    chex.assert_equal_shape((predictions, targets))
    errors = predictions - targets
    return jnp.square(errors)


def _l1_loss(predictions, targets):
    chex.assert_type([predictions], float)
    chex.assert_equal_shape((predictions, targets))
    errors = predictions - targets
    return jnp.abs(errors)
