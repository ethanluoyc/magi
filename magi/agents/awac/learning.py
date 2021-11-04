"""
# Referred to
# https://github.com/ikostrikov/jaxrl/blob/4e42ff2bbffe9af8027ec77bd557c31c80c83e9b/jaxrl/agents/awac/awac_learner.py
"""
import copy
import functools
from typing import Iterator, NamedTuple, Optional

from acme import core
from acme import specs
from acme import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb


def mse_loss(a, b):
    return jnp.mean(jnp.square(a - b))


class TrainingState(NamedTuple):
    policy_params: hk.Params
    critic_params: hk.Params
    policy_opt_state: hk.Params
    critic_opt_state: hk.Params
    critic_target_params: hk.Params


class AWACLearner(core.Learner):
    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy_network: hk.Transformed,
        critic_network: hk.Transformed,
        iterator: Iterator[reverb.ReplaySample],
        random_key: jnp.ndarray,
        policy_optimizer: Optional[optax.GradientTransformation] = None,
        critic_optimizer: Optional[optax.GradientTransformation] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 2,
        beta: float = 1.0,
        num_samples: int = 1,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):
        self._policy_optimizer = (
            policy_optimizer if policy_optimizer else optax.adam(3e-4)
        )
        self._critic_optimizer = (
            critic_optimizer if critic_optimizer else optax.adam(3e-4)
        )
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._rng = hk.PRNGSequence(random_key)
        self._data_iterator = iterator
        self._discount = discount
        self._tau = tau
        self._target_update_period = target_update_period
        self._learning_steps = 0
        self._environment_spec = environment_spec
        self.beta = beta
        self.num_samples = num_samples

        def init_state():
            dummy_obs = utils.add_batch_dim(
                utils.zeros_like(environment_spec.observations)
            )
            dummy_actions = utils.add_batch_dim(
                utils.zeros_like(environment_spec.actions)
            )
            policy_params = policy_network.init(next(self._rng), dummy_obs)
            policy_opt_state = self._policy_optimizer.init(policy_params)
            critic_params = critic_network.init(
                next(self._rng), dummy_obs, dummy_actions
            )
            critic_opt_state = self._policy_optimizer.init(critic_params)
            return TrainingState(
                policy_params=policy_params,
                critic_params=critic_params,
                policy_opt_state=policy_opt_state,
                critic_opt_state=critic_opt_state,
                critic_target_params=copy.deepcopy(critic_params),
            )

        self._state = init_state()
        self._logger = logger or loggers.make_default_logger("learner", save_data=False)
        self._counter = counter or counting.Counter()

        @functools.partial(jax.jit, static_argnums=(5, 6))
        def _update_actor(
            actor_params,
            opt_state,
            key,
            critic_params,
            batch: reverb.ReplaySample,
            num_samples: int,
            beta: float,
        ):
            transitions: types.ReplaySample = batch.data

            v1, v2 = get_value(
                key, actor_params, critic_params, transitions.observation, num_samples
            )
            v = jnp.minimum(v1, v2)

            def actor_loss_fn(actor_params: hk.Params):
                dist = self._policy_network.apply(actor_params, transitions.observation)
                # TODO(yl): move limit handling to dataset processing
                lim = 1 - 1e-5
                actions = jnp.clip(transitions.action, -lim, lim)
                # Note: in AWAC, the actions come from the replay buffer
                log_probs = dist.log_prob(actions)

                q1, q2 = self._critic_network.apply(
                    critic_params, transitions.observation, actions
                )
                q = jnp.minimum(q1, q2)
                a = q - v

                # we could have used exp(a / beta) here but
                # exp(a / beta) is unbiased but high variance,
                # softmax(a / beta) is biased but lower variance.
                # sum() instead of mean(), because it should be multiplied by batch size.
                actor_loss = -(jax.nn.softmax(a / beta) * log_probs).sum(axis=0)

                return actor_loss, {"policy_loss": actor_loss}

            (_, metrics), grad = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                actor_params
            )
            update, opt_state = self._policy_optimizer.update(grad, opt_state)
            params_actor = optax.apply_updates(actor_params, update)
            return params_actor, opt_state, metrics

        def get_value(key, actor_params, critic_params, observations, num_samples: int):
            dist = self._policy_network.apply(actor_params, observations)

            policy_actions = dist.sample(seed=key, sample_shape=(num_samples,))

            n_observations = jnp.repeat(observations[jnp.newaxis], num_samples, axis=0)
            q_pi1, q_pi2 = self._critic_network.apply(
                critic_params, n_observations, policy_actions
            )

            def get_v(q):
                return jnp.mean(q, axis=0)

            return get_v(q_pi1), get_v(q_pi2)

        @jax.jit
        def _update_critic(
            params_critic,
            opt_state,
            key,
            critic_target_params,
            actor_params,
            batch: reverb.ReplaySample,
        ):
            transitions: types.Transition = batch.data

            def loss_fn(critic_params):
                next_action_dist = self._policy_network.apply(
                    actor_params, transitions.next_observation
                )
                next_action = next_action_dist.sample(seed=key)

                # Compute the target Q value
                q1_target, q2_target = self._critic_network.apply(
                    critic_target_params, transitions.next_observation, next_action
                )
                q_target = jnp.minimum(q1_target, q2_target)
                # Note: For now the temperature is removed as per AWAC
                q_target = jax.lax.stop_gradient(
                    transitions.reward
                    + transitions.discount * self._discount * q_target
                )

                # Get current Q estimates
                q1, q2 = self._critic_network.apply(
                    critic_params, transitions.observation, transitions.action
                )

                critic_loss = mse_loss(q1, q_target) + mse_loss(q2, q_target)
                return critic_loss

            loss, grad = jax.value_and_grad(loss_fn)(params_critic)
            update, opt_state = self._critic_optimizer.update(grad, opt_state)
            params_critic = optax.apply_updates(params_critic, update)
            return params_critic, opt_state, {"critic_loss": loss}

        @jax.jit
        def _update_target(state: TrainingState):
            return state._replace(
                critic_target_params=optax.incremental_update(
                    state.critic_params,
                    state.critic_target_params,
                    tau,
                ),
            )

        self._update_actor = _update_actor
        self._update_critic = _update_critic
        self._update_target = _update_target

    def step(self):
        # Sample replay buffer
        self._learning_steps += 1

        # Sample replay buffer
        batch = next(self._data_iterator)

        critic_params, critic_opt_state, critic_metrics = self._update_critic(
            self._state.critic_params,
            self._state.critic_opt_state,
            next(self._rng),
            self._state.critic_target_params,
            self._state.policy_params,
            batch,
        )
        self._state = self._state._replace(
            critic_params=critic_params, critic_opt_state=critic_opt_state
        )

        policy_params, policy_opt_state, policy_metrics = self._update_actor(
            self._state.policy_params,
            self._state.policy_opt_state,
            next(self._rng),
            self._state.critic_params,
            batch,
            self.num_samples,
            self.beta,
        )
        self._state = self._state._replace(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
        )

        # Update target params
        if self._learning_steps % self._target_update_period == 0:
            # Update frozen target models
            self._state = self._update_target(self._state)

        counts = self._counter.increment(steps=1)
        metrics = {**policy_metrics, **critic_metrics, **counts}

        self._logger.write(metrics)

    def get_variables(self, names):
        del names
        return [self._state.policy_params]
