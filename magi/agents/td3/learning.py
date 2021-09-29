"""Learner for Twin-Delayed DDPG agent."""
import copy
from typing import Iterator, NamedTuple, Optional

import acme
from acme import types
from acme.jax import types as jax_types
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb


def _mse_loss(a, b):
    return jnp.mean(jnp.square(a - b))


class TrainingState(NamedTuple):
    policy_params: hk.Params
    critic_params: hk.Params
    policy_opt_state: hk.Params
    critic_opt_state: hk.Params
    policy_target_params: hk.Params
    critic_target_params: hk.Params
    key: jax_types.PRNGKey


class TD3Learner(acme.Learner):
    def __init__(
        self,
        policy_network: hk.Transformed,
        critic_network: hk.Transformed,
        iterator: Iterator[reverb.ReplaySample],
        random_key: jnp.ndarray,
        policy_optimizer: Optional[optax.GradientTransformation] = None,
        critic_optimizer: Optional[optax.GradientTransformation] = None,
        discount: float = 0.99,
        soft_update_rate: float = 0.005,
        policy_noise: float = 0.2,
        policy_noise_clip: float = 0.5,
        policy_target_update_period: int = 2,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):

        # Internalize parameters
        self._policy_optimizer = (
            policy_optimizer if policy_optimizer else optax.adam(3e-4)
        )
        self._critic_optimizer = (
            critic_optimizer if critic_optimizer else optax.adam(3e-4)
        )
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._data_iterator = iterator
        self._discount = discount
        self._soft_update_rate = soft_update_rate
        self._policy_noise = policy_noise
        self._policy_noise_clip = policy_noise_clip
        self._policy_target_update_period = policy_target_update_period
        self._learning_steps = 0
        self._logger = logger or loggers.make_default_logger("learner", save_data=False)
        self._counter = counter or counting.Counter()

        def init_state(key):
            key1, key2, key = jax.random.split(key, 3)
            policy_params = policy_network.init(key1)
            policy_opt_state = self._policy_optimizer.init(policy_params)
            critic_params = critic_network.init(key2)
            critic_opt_state = self._policy_optimizer.init(critic_params)
            return TrainingState(
                policy_params,
                critic_params,
                policy_opt_state,
                critic_opt_state,
                copy.deepcopy(policy_params),
                copy.deepcopy(critic_params),
                key=key,
            )

        self._state = init_state(random_key)

        @jax.jit
        def _update_policy(state: TrainingState, batch: reverb.ReplaySample, key):
            del key

            policy_params = state.policy_params
            policy_opt_state = state.policy_opt_state
            critic_params = state.critic_params

            transitions: types.Transition = batch.data

            def loss_fn(policy_params):
                a_t = self._policy_network.apply(policy_params, transitions.observation)
                # TD3 uses the q values from the first critic for updating the actor
                q, _ = self._critic_network.apply(
                    critic_params, transitions.observation, a_t
                )
                policy_loss = -jnp.mean(q, axis=0)
                return policy_loss

            loss, grads = jax.value_and_grad(loss_fn)(policy_params)
            update, new_policy_opt_state = self._policy_optimizer.update(
                grads, policy_opt_state
            )
            new_policy_params = optax.apply_updates(policy_params, update)
            new_state = state._replace(
                policy_params=new_policy_params, policy_opt_state=new_policy_opt_state
            )
            return new_state, {"policy_loss": loss}

        @jax.jit
        def _update_critic(
            state: TrainingState, batch: reverb.ReplaySample, key: jax_types.PRNGKey
        ):
            transitions: types.Transition = batch.data

            critic_params = state.critic_params
            critic_opt_state = state.critic_opt_state
            policy_target_params = state.policy_target_params
            critic_target_params = state.critic_target_params

            def loss_fn(critic_params):
                # Select action according to policy and add clipped noise
                noise = jnp.clip(
                    jax.random.normal(key, transitions.action.shape)
                    * self._policy_noise,
                    -self._policy_noise_clip,
                    self._policy_noise_clip,
                )

                next_action = self._policy_network.apply(
                    policy_target_params, transitions.next_observation
                )
                next_action = next_action + noise
                next_action = jnp.clip(next_action, -1.0, 1.0)

                # Compute the target Q value
                q1_target, q2_target = self._critic_network.apply(
                    critic_target_params, transitions.next_observation, next_action
                )
                q_target = jnp.minimum(q1_target, q2_target)
                q_target = jax.lax.stop_gradient(
                    transitions.reward + discount * self._discount * q_target
                )

                # Get current Q estimates
                q1, q2 = self._critic_network.apply(
                    critic_params, transitions.observation, transitions.action
                )

                q1_loss = _mse_loss(q1, q_target)
                q2_loss = _mse_loss(q2, q_target)

                critic_loss = q1_loss + q2_loss
                return critic_loss

            loss, grads = jax.value_and_grad(loss_fn)(critic_params)
            updates, new_critic_opt_state = self._critic_optimizer.update(
                grads, critic_opt_state
            )
            new_critic_params = optax.apply_updates(critic_params, updates)
            new_state = state._replace(
                critic_params=new_critic_params, critic_opt_state=new_critic_opt_state
            )
            return new_state, {"critic_loss": loss}

        @jax.jit
        def _update_target(state: TrainingState):
            return state._replace(
                policy_target_params=optax.incremental_update(
                    state.policy_params,
                    state.policy_target_params,
                    soft_update_rate,
                ),
                critic_target_params=optax.incremental_update(
                    state.critic_params,
                    state.critic_target_params,
                    soft_update_rate,
                ),
            )

        def sgd_step(state: TrainingState, batch: reverb.ReplaySample, step: int):
            metrics = {}
            key1, key2, key = jax.random.split(state.key, 3)
            state, critic_metrics = _update_critic(state, batch, key1)
            metrics.update(critic_metrics)

            # Delayed policy and target network updates
            if step % policy_target_update_period == 0:
                state, policy_metrics = _update_policy(
                    state,
                    batch,
                    key2,
                )
                metrics.update(policy_metrics)
                # Update target networks
                state = _update_target(state)
            state = state._replace(key=key)
            return state, metrics

        self._sgd_step = sgd_step

    def step(self):
        # Sample replay buffer
        batch = next(self._data_iterator)

        self._state, metrics = self._sgd_step(self._state, batch, self._learning_steps)
        self._learning_steps += 1
        counts = self._counter.increment(steps=1)
        self._logger.write({**counts, **metrics})

    def get_variables(self, names):
        variables = {
            "policy": self._state.policy_params,
            "critic": self._state.critic_params,
        }
        return [variables[name] for name in names]

    def restore(self, state):
        self._state = state

    def save(self):
        return self._state
