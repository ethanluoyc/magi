"""SAC Learner."""
from functools import partial
import time
from typing import Iterator, NamedTuple, Optional

from acme import core
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb

from magi.agents.sac import losses


class TrainingState(NamedTuple):
    """Training state for SAC learner."""

    policy_params: networks_lib.Params
    critic_params: networks_lib.Params
    critic_target_params: networks_lib.Params
    policy_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    log_alpha: jnp.ndarray
    key: jax_types.PRNGKey


class SACLearner(core.Learner):
    """SAC learner."""

    def __init__(
        self,
        policy: networks_lib.FeedForwardNetwork,
        critic: networks_lib.FeedForwardNetwork,
        random_key: jax_types.PRNGKey,
        dataset: Iterator[reverb.ReplaySample],
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        alpha_optimizer: optax.GradientTransformation,
        target_entropy: float,
        discount: float = 0.99,
        tau: float = 5e-3,
        init_alpha: float = 1.0,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):
        self._iterator = dataset

        self._logger = (
            logger
            if logger is not None
            else loggers.make_default_logger(label="learner", save_data=False)
        )
        self._counter = counter if counter is not None else counting.Counter()

        def init_state(key):
            init_policy_key, init_critic_key, key = jax.random.split(random_key, 3)
            # Actor.
            init_policy_params = policy.init(init_policy_key)
            init_critic_params = critic.init(init_critic_key)
            init_policy_optimizer_state = actor_optimizer.init(init_policy_params)
            init_critic_optimizer_state = critic_optimizer.init(init_critic_params)
            init_log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
            init_alpha_optimizer_state = alpha_optimizer.init(init_log_alpha)
            return TrainingState(
                policy_params=init_policy_params,
                critic_params=init_critic_params,
                critic_target_params=init_critic_params,
                policy_optimizer_state=init_policy_optimizer_state,
                critic_optimizer_state=init_critic_optimizer_state,
                alpha_optimizer_state=init_alpha_optimizer_state,
                log_alpha=init_log_alpha,
                key=key,
            )

        self._state = init_state(random_key)

        @jax.jit
        def _update_actor(state: TrainingState, transitions: types.Transition):
            def loss_fn(policy_params, key):
                return losses.actor_loss_fn(
                    policy,
                    critic,
                    policy_params,
                    key,
                    state.critic_params,
                    state.log_alpha,
                    transitions.observation,
                )

            step_key, key = jax.random.split(state.key)
            (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                state.policy_params, step_key
            )
            update, policy_optimizer_state = actor_optimizer.update(
                grad, state.policy_optimizer_state
            )
            policy_params = optax.apply_updates(state.policy_params, update)
            state = state._replace(
                policy_params=policy_params,
                policy_optimizer_state=policy_optimizer_state,
                key=key,
            )
            return (state, loss, aux)

        @jax.jit
        def _update_critic(
            state: TrainingState,
            transitions: types.Transition,
        ):
            def loss_fn(critic_params, key):
                return losses.critic_loss_fn(
                    policy,
                    critic,
                    critic_params,
                    key,
                    state.critic_target_params,
                    state.policy_params,
                    state.log_alpha,
                    transitions,
                    gamma=discount,
                )

            step_key, key = jax.random.split(state.key)
            (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                state.critic_params, step_key
            )
            update, critic_optimizer_state = critic_optimizer.update(
                grad, state.critic_optimizer_state
            )
            critic_params = optax.apply_updates(state.critic_params, update)
            state = state._replace(
                critic_params=critic_params,
                critic_optimizer_state=critic_optimizer_state,
                key=key,
            )
            return state, loss, aux

        @jax.jit
        def _update_alpha(state: TrainingState, entropy: jnp.ndarray):
            def loss_fn(log_alpha):
                return losses.alpha_loss_fn(
                    log_alpha, entropy, target_entropy=target_entropy
                )

            (loss, _), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.log_alpha)
            update, alpha_optimizer_state = alpha_optimizer.update(
                grad, state.alpha_optimizer_state
            )
            log_alpha = optax.apply_updates(state.log_alpha, update)
            state = state._replace(
                log_alpha=log_alpha, alpha_optimizer_state=alpha_optimizer_state
            )
            return state, loss, {"alpha": jnp.exp(log_alpha)}

        @jax.jit
        def _update_target(state: TrainingState):
            update_fn = partial(optax.incremental_update, step_size=tau)
            return state._replace(
                critic_target_params=update_fn(
                    state.critic_params, state.critic_target_params
                )
            )

        def sgd_step(state: TrainingState, transitions: types.Transition):
            state, critic_loss, critic_metrics = _update_critic(state, transitions)
            state, actor_loss, actor_metrics = _update_actor(state, transitions)
            entropy = actor_metrics["entropy"]
            state, alpha_loss, alpha_metrics = _update_alpha(state, entropy)
            state = _update_target(state)
            metrics = {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                **critic_metrics,
                **actor_metrics,
                **alpha_metrics,
            }
            return state, metrics

        self._sgd_step = sgd_step
        self._timestamp = None

    def step(self):
        # Get data from replay
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
            "policy": self._state.policy_params,
            "critic": self._state.critic_params,
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState):
        self._state = state
