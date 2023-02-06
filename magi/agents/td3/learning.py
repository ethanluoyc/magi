"""Learner for Twin-Delayed DDPG agent."""
import time
from typing import Iterator, NamedTuple, Optional

import acme
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers


def _mse_loss(a, b):
    return jnp.mean(jnp.square(a - b))


class TrainingState(NamedTuple):
    """Training state for TD3 learner."""

    policy_params: hk.Params
    critic_params: hk.Params
    policy_opt_state: hk.Params
    critic_opt_state: hk.Params
    policy_target_params: hk.Params
    critic_target_params: hk.Params
    key: jax_types.PRNGKey
    steps: int


class TD3Learner(acme.Learner):
    """TD3 learner."""

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
        if not critic_optimizer:
            critic_optimizer = optax.adam(3e-4)
        if not policy_optimizer:
            policy_optimizer = optax.adam(3e-4)

        def policy_loss_fn(
            policy_params: networks_lib.Params,
            critic_params: networks_lib.Params,
            transitions: types.Transition,
        ):
            a_t = policy_network.apply(policy_params, transitions.observation)
            # TD3 uses the q values from the first critic for updating the actor
            q, _ = critic_network.apply(critic_params, transitions.observation, a_t)
            policy_loss = -jnp.mean(q, axis=0)
            return policy_loss

        def critic_loss_fn(
            critic_params: networks_lib.Params,
            policy_target_params: networks_lib.Params,
            critic_target_params: networks_lib.Params,
            transitions: types.Transition,
            random_key: jax_types.PRNGKey,
        ):
            # Select action according to policy and add clipped noise
            noise = jnp.clip(
                jax.random.normal(random_key, transitions.action.shape) * policy_noise,
                -policy_noise_clip,
                policy_noise_clip,
            )

            next_action = policy_network.apply(
                policy_target_params, transitions.next_observation
            )
            next_action = next_action + noise
            next_action = jnp.clip(next_action, -1.0, 1.0)

            # Compute the target Q value
            q1_target, q2_target = critic_network.apply(
                critic_target_params, transitions.next_observation, next_action
            )
            q_target = jnp.minimum(q1_target, q2_target)
            q_target = jax.lax.stop_gradient(
                transitions.reward + transitions.discount * discount * q_target
            )

            # Get current Q estimates
            q1, q2 = critic_network.apply(
                critic_params, transitions.observation, transitions.action
            )

            q1_loss = _mse_loss(q1, q_target)
            q2_loss = _mse_loss(q2, q_target)

            critic_loss = q1_loss + q2_loss
            return critic_loss, {"q1": jnp.mean(q1), "q2": jnp.mean(q2)}

        def sgd_step(state: TrainingState, batch: reverb.ReplaySample):
            soft_update = lambda p, tp: optax.incremental_update(
                p, tp, soft_update_rate
            )
            critic_key, key = jax.random.split(state.key)
            # Update critic
            (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(
                state.critic_params,
                state.policy_target_params,
                state.critic_target_params,
                batch,
                critic_key,
            )

            critic_updates, critic_opt_state = critic_optimizer.update(
                critic_grads, state.critic_opt_state
            )
            critic_params = optax.apply_updates(state.critic_params, critic_updates)

            policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(
                state.policy_params, state.critic_params, batch
            )

            def update_policy_state():
                policy_updates, policy_opt_state = policy_optimizer.update(
                    policy_grads, state.policy_opt_state
                )
                policy_params = optax.apply_updates(state.policy_params, policy_updates)
                policy_target_params = soft_update(
                    policy_params, state.policy_target_params
                )
                return policy_params, policy_target_params, policy_opt_state

            # The update on the policy and target critic is applied every
            # `update_period` steps.
            current_policy_state = (
                state.policy_params,
                state.policy_target_params,
                state.policy_opt_state,
            )
            policy_params, policy_target_params, policy_opt_state = jax.lax.cond(
                state.steps % policy_target_update_period == 0,
                lambda _: update_policy_state(),
                lambda _: current_policy_state,
                operand=None,
            )
            # In the original implementation the critic target updates are also delayed.
            critic_target_params = jax.lax.cond(
                state.steps % policy_target_update_period == 0,
                lambda _: soft_update(critic_params, state.critic_target_params),
                lambda _: state.critic_target_params,
                operand=None,
            )
            steps = state.steps + 1
            state = TrainingState(
                policy_params=policy_params,
                critic_params=critic_params,
                policy_opt_state=policy_opt_state,
                critic_opt_state=critic_opt_state,
                policy_target_params=policy_target_params,
                critic_target_params=critic_target_params,
                key=key,
                steps=steps,
            )
            metrics = {
                **critic_metrics,
                "policy_loss": policy_loss,
                "critic_loss": critic_loss,
            }
            return state, metrics

        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            "learner",
            save_data=False,
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
        )

        self._sgd_step = jax.jit(sgd_step)

        self._iterator = iterator

        def make_initial_state(key: jax_types.PRNGKey):
            key1, key2, key = jax.random.split(key, 3)
            policy_params = policy_network.init(key1)
            policy_opt_state = policy_optimizer.init(policy_params)
            critic_params = critic_network.init(key2)
            critic_opt_state = critic_optimizer.init(critic_params)
            return TrainingState(
                policy_params=policy_params,
                critic_params=critic_params,
                policy_opt_state=policy_opt_state,
                critic_opt_state=critic_opt_state,
                policy_target_params=policy_params,
                critic_target_params=critic_params,
                key=key,
                steps=0,
            )

        self._state = make_initial_state(random_key)
        self._timestamp = None

    def step(self):
        # Sample replay buffer
        batch = next(self._iterator).data

        self._state, metrics = self._sgd_step(self._state, batch)

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
            "policy": self._state.policy_params,
            "critic": self._state.critic_params,
        }
        return [variables[name] for name in names]

    def restore(self, state: TrainingState):
        self._state = state

    def save(self):
        return self._state
