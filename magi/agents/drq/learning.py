"""Learner component for DrQ."""
import functools
import time
from typing import Iterator, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
from acme import core
from acme import types
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers

from magi.agents.drq import augmentations


def _sample_and_log_prob(dist, key):
    if hasattr(dist, "sample_and_log_prob"):
        # Support Distrax sample_and_log_prob shortcut
        actions, log_probs = dist.sample_and_log_prob(seed=key)
    else:
        # Fall back to calling sample and log_prob separately
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
    return actions, log_probs


def _critic_loss_fn(encoder_fn, policy_fn, critic_fn, gamma: float):
    def policy(params, obs, key):
        encoded = jax.lax.stop_gradient(encoder_fn(params["encoder"], obs))
        dist = policy_fn(params["policy"], encoded)
        return _sample_and_log_prob(dist, key)

    def critic(params, obs, action):
        encoded = encoder_fn(params["encoder"], obs)
        return critic_fn(params["critic"], encoded, action)

    def loss_fn(
        critic_params: networks_lib.Params,
        target_critic_params: networks_lib.Params,
        policy_params: networks_lib.Params,
        alpha: jnp.ndarray,
        data: types.Transition,
        key: networks_lib.PRNGKey,
    ):
        # In SAC, the next action is computed using the online policy parameters,
        # which means that the online critic params should be used for sampling
        # a_t+1
        next_actions, next_log_probs = policy(
            {"policy": policy_params, "encoder": critic_params["encoder"]},
            data.next_observation,
            key,
        )

        # Target critic params still used for computing q_t, so using the
        # target encoder here as well.
        next_q1, next_q2 = critic(
            target_critic_params, data.next_observation, next_actions
        )
        next_q = jnp.minimum(next_q1, next_q2)
        next_q -= alpha * next_log_probs
        target_q = data.reward + data.discount * gamma * next_q
        target_q = jax.lax.stop_gradient(target_q)
        # Calculate predicted Q
        q1, q2 = critic(critic_params, data.observation, data.action)
        chex.assert_rank(
            (next_log_probs, target_q, next_q1, next_q2, q1, q2), [1, 1, 1, 1, 1, 1]
        )
        loss_critic = (jnp.square(target_q - q1) + jnp.square(target_q - q2)).mean(
            axis=0
        )
        return loss_critic, {"q1": q1.mean(), "q2": q2.mean()}

    return loss_fn


def _actor_loss_fn(encoder_fn, policy_fn, critic_fn):
    def loss_fn(
        policy_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        alpha: jnp.ndarray,
        transitions: types.Transition,
        key: networks_lib.PRNGKey,
    ):
        encoded = encoder_fn(critic_params["encoder"], transitions.observation)
        action_dist = policy_fn(policy_params, encoded)
        actions, log_probs = _sample_and_log_prob(action_dist, key)
        q1, q2 = critic_fn(critic_params["critic"], encoded, actions)
        chex.assert_rank((q1, q2, log_probs), [1, 1, 1])
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * alpha - q).mean(axis=0)
        entropy = -log_probs.mean()
        return actor_loss, {"entropy": entropy}

    return loss_fn


def _alpha_loss_fn(log_alpha: jnp.ndarray, entropy: jnp.ndarray, target_entropy: float):
    temperature = jnp.exp(log_alpha)
    return temperature * (entropy - target_entropy), {"alpha": temperature}


class TrainingState(NamedTuple):
    """Holds training state for the DrQ learner."""

    policy_params: networks_lib.Params
    encoder_params: networks_lib.Params
    critic_params: networks_lib.Params
    target_encoder_params: networks_lib.Params
    target_critic_params: networks_lib.Params

    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState

    log_alpha_params: jnp.ndarray
    alpha_opt_state: jnp.ndarray
    key: networks_lib.PRNGKey
    steps: int

    @property
    def encoder_critic_params(self):
        return {
            "encoder": self.encoder_params,
            "critic": self.critic_params,
        }

    @property
    def encoder_critic_target_params(self):
        return {
            "encoder": self.target_encoder_params,
            "critic": self.target_critic_params,
        }


class DrQLearner(core.Learner):
    """Learner for Data-regularized Q"""

    def __init__(
        self,
        random_key: jnp.ndarray,
        dataset: Iterator[reverb.ReplaySample],
        encoder_network: networks_lib.FeedForwardNetwork,
        policy_network: networks_lib.FeedForwardNetwork,
        critic_network: networks_lib.FeedForwardNetwork,
        target_entropy: float,
        augmentation: Optional[augmentations.DataAugmentation] = None,
        policy_optimizer: Optional[optax.GradientTransformation] = None,
        critic_optimizer: Optional[optax.GradientTransformation] = None,
        temperature_optimizer: Optional[optax.GradientTransformation] = None,
        init_temperature: float = 0.1,
        actor_update_frequency: int = 1,
        critic_target_update_frequency: int = 1,
        critic_soft_update_rate: float = 0.005,
        discount: float = 0.99,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        # pytype: disable=attribute-error
        policy_optimizer = policy_optimizer or optax.adam(1e-4)
        critic_optimizer = critic_optimizer or optax.adam(1e-4)
        alpha_optimizer = temperature_optimizer or optax.adam(3e-4)
        augmentation = augmentation or augmentations.batched_random_crop
        augmentation: augmentations.DataAugmentation = jax.jit(augmentation)

        # Setup losses
        critic_loss_fn = _critic_loss_fn(
            encoder_network.apply,
            policy_network.apply,
            critic_network.apply,
            gamma=discount,
        )
        actor_loss_fn = _actor_loss_fn(
            encoder_network.apply, policy_network.apply, critic_network.apply
        )
        alpha_loss_fn = functools.partial(_alpha_loss_fn, target_entropy=target_entropy)

        critic_loss_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        actor_loss_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        alpha_loss_grad_fn = jax.value_and_grad(alpha_loss_fn, has_aux=True)
        polyak_average = functools.partial(
            optax.incremental_update, step_size=critic_soft_update_rate
        )

        def sgd_step(
            state: TrainingState,
            transitions: types.Transition,
        ):
            key_aug, key_aug_next, key_actor, key_critic, key = jax.random.split(
                state.key, 5
            )
            alpha = jnp.exp(state.log_alpha_params)

            # Perform data augmentation on o_tm1 and o_t
            transitions = transitions._replace(
                observation=augmentation(key_aug, transitions.observation),
                next_observation=augmentation(
                    key_aug_next, transitions.next_observation
                ),
            )

            (critic_loss, critic_metrics), critic_grads = critic_loss_grad_fn(
                state.encoder_critic_params,
                state.encoder_critic_target_params,
                state.policy_params,
                alpha,
                transitions,
                key_critic,
            )
            critic_update, critic_opt_state = critic_optimizer.update(
                critic_grads, state.critic_opt_state
            )
            new_params = optax.apply_updates(state.encoder_critic_params, critic_update)
            critic_params = new_params["critic"]
            encoder_params = new_params["encoder"]

            (actor_loss, actor_metrics), actor_grads = actor_loss_grad_fn(
                state.policy_params,
                state.encoder_critic_params,
                alpha,
                transitions,
                key_actor,
            )
            (alpha_loss, alpha_metrics), alpha_grads = alpha_loss_grad_fn(
                state.log_alpha_params, actor_metrics["entropy"]
            )
            # Update
            steps = state.steps + 1

            def update_policy_step():
                actor_update, policy_opt_state = policy_optimizer.update(
                    actor_grads, state.policy_opt_state
                )
                policy_params = optax.apply_updates(state.policy_params, actor_update)
                alpha_updates, alpha_opt_state = alpha_optimizer.update(
                    alpha_grads, state.alpha_opt_state
                )
                log_alpha = optax.apply_updates(state.log_alpha_params, alpha_updates)
                return policy_params, policy_opt_state, log_alpha, alpha_opt_state

            (
                policy_params,
                policy_opt_state,
                log_alpha_params,
                alpha_opt_state,
            ) = jax.lax.cond(
                steps % actor_update_frequency == 0,
                lambda _: update_policy_step(),
                lambda _: (
                    state.policy_params,
                    state.policy_opt_state,
                    state.log_alpha_params,
                    state.alpha_opt_state,
                ),
                operand=None,
            )

            online_params = (critic_params, encoder_params)
            target_params = (state.target_critic_params, state.target_encoder_params)
            target_critic_params, target_encoder_params = jax.lax.cond(
                steps % critic_target_update_frequency == 0,
                lambda _: polyak_average(online_params, target_params),
                lambda _: target_params,
                operand=None,
            )
            metrics = {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "alpha_loss": alpha_loss,
                **actor_metrics,
                **critic_metrics,
                **alpha_metrics,
            }
            state = TrainingState(
                policy_params=policy_params,
                encoder_params=encoder_params,
                critic_params=critic_params,
                target_encoder_params=target_encoder_params,
                target_critic_params=target_critic_params,
                policy_opt_state=policy_opt_state,
                critic_opt_state=critic_opt_state,
                log_alpha_params=log_alpha_params,
                alpha_opt_state=alpha_opt_state,
                key=key,
                steps=steps,
            )
            return state, metrics

        def make_initial_state(key):
            # Initialize training state
            key1, key2, key3, key = jax.random.split(key, 4)
            encoder_init_params = encoder_network.init(key1)
            actor_init_params = policy_network.init(key3)
            critic_init_params = critic_network.init(key2)
            encoder_init_target_params = encoder_init_params
            critic_init_target_params = critic_init_params
            # Entropy coefficient.
            init_log_alpha = jnp.array(np.log(init_temperature), dtype=jnp.float32)

            actor_init_opt_state = policy_optimizer.init(actor_init_params)
            encoder_init_critic_opt_state = critic_optimizer.init(
                {"encoder": encoder_init_params, "critic": critic_init_params}
            )
            alpha_init_opt_state = alpha_optimizer.init(init_log_alpha)

            return TrainingState(
                policy_params=actor_init_params,
                encoder_params=encoder_init_params,
                critic_params=critic_init_params,
                target_encoder_params=encoder_init_target_params,
                target_critic_params=critic_init_target_params,
                policy_opt_state=actor_init_opt_state,
                critic_opt_state=encoder_init_critic_opt_state,
                log_alpha_params=init_log_alpha,
                alpha_opt_state=alpha_init_opt_state,
                key=key,
                steps=0,
            )

        self._state = make_initial_state(random_key)
        self._iterator = dataset

        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            label="learner", save_data=False
        )

        self._timestamp = None
        self._sgd_step = jax.jit(sgd_step)
        # pytype: enable=attribute-error

    def step(self):
        # Get the next batch from the replay iterator
        sample = next(self._iterator)
        transitions: types.Transition = sample.data

        # Perform a single learner step
        self._state, metrics = self._sgd_step(self._state, transitions)

        # Compute elapsed time
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp
        # Increment counts and record the current time
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        # Attempts to write the logs.
        self._logger.write({**metrics, **counts})

    def get_variables(self, names):
        variables = {
            "policy": {
                "encoder": self._state.encoder_params,
                "actor": self._state.policy_params,
            },
        }
        return [variables[name] for name in names]

    def save(self):
        return self._state

    def restore(self, state):
        self._state = state
