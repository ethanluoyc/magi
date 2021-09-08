"""Learner for Twin-Delayed DDPG agent."""
import copy
from typing import Iterator, NamedTuple, Optional

import acme
from acme import specs
from acme import types
from acme.jax import types as jax_types
from acme.jax import utils
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


class TD3Learner(acme.Learner):
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
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_update_period: int = 2,
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
        self._rng = hk.PRNGSequence(random_key)
        self._action_spec: specs.BoundedArray = environment_spec.actions
        self._data_iterator = iterator
        self._discount = discount
        self._tau = tau
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._policy_update_period = policy_update_period
        self._learning_steps = 0
        self._logger = logger or loggers.make_default_logger("learner", save_data=False)
        self._counter = counter or counting.Counter()

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
                policy_params,
                critic_params,
                policy_opt_state,
                critic_opt_state,
                copy.deepcopy(policy_params),
                copy.deepcopy(critic_params),
            )

        self._state = init_state()

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
                policy_loss = jnp.mean(q, axis=0)
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
                    -self._noise_clip,
                    self._noise_clip,
                )

                next_action = self._policy_network.apply(
                    policy_target_params, transitions.next_observation
                )
                next_action = next_action + noise
                next_action = jnp.clip(
                    next_action, self._action_spec.minimum, self._action_spec.maximum
                )

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
                    tau,
                ),
                critic_target_params=optax.incremental_update(
                    state.critic_params,
                    state.critic_target_params,
                    tau,
                ),
            )

        self._update_policy = _update_policy
        self._update_critic = _update_critic
        self._update_target = _update_target

    def step(self):
        self._learning_steps += 1

        # Sample replay buffer
        batch = next(self._data_iterator)

        metrics = {}
        self._state, critic_metrics = self._update_critic(
            self._state, batch, next(self._rng)
        )
        metrics.update(critic_metrics)

        # Delayed policy and target network updates
        if self._learning_steps % self._policy_update_period == 0:
            self._state, policy_metrics = self._update_policy(
                self._state,
                batch,
                next(self._rng),
            )
            metrics.update(policy_metrics)
            # Update target networks
            self._state = self._update_target(self._state)
        counts = self._counter.increment(steps=1)
        metrics.update(counts)

        self._logger.write(metrics)

    def get_variables(self, names):
        del names
        return [self._state.policy_params]
