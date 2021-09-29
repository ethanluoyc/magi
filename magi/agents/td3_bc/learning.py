"V1 does not work on hopper-medium, investigate why"
import copy
from typing import Iterator, List, NamedTuple, Optional

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
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
    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    policy_target_params: hk.Params
    critic_target_params: hk.Params
    key: jax_types.PRNGKey


class TD3BCLearner(acme.Learner):
    def __init__(
        self,
        policy_network: networks_lib.FeedForwardNetwork,
        critic_network: networks_lib.FeedForwardNetwork,
        iterator: Iterator[reverb.ReplaySample],
        random_key: jax_types.PRNGKey,
        policy_optimizer: Optional[optax.GradientTransformation] = None,
        critic_optimizer: Optional[optax.GradientTransformation] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_update_period: int = 2,
        alpha: float = 2.5,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):

        self._data_iterator = iterator
        self._policy_optimizer = policy_optimizer or optax.adam(3e-4)
        self._critic_optimizer = critic_optimizer or optax.adam(3e-4)
        self._discount = discount
        self._tau = tau
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._policy_update_period = policy_update_period
        self._alpha = alpha

        policy_init_key, critic_init_key, random_key = jax.random.split(random_key, 3)
        policy_init_params = policy_network.init(policy_init_key)
        policy_init_opt_state = self._policy_optimizer.init(policy_init_params)
        critic_init_params = critic_network.init(critic_init_key)
        critic_init_opt_state = self._critic_optimizer.init(critic_init_params)
        self._state = TrainingState(
            policy_init_params,
            critic_init_params,
            policy_init_opt_state,
            critic_init_opt_state,
            copy.deepcopy(policy_init_params),
            copy.deepcopy(critic_init_params),
            random_key,
        )

        self._learning_steps = 0
        self._logger = logger or loggers.make_default_logger("learner")
        self._counter = counter or counting.Counter()

        @jax.jit
        def _update_actor(state: TrainingState, transitions: types.Transition):
            def loss_fn(policy_params):
                pi = policy_network.apply(policy_params, transitions.observation)
                q, _ = critic_network.apply(
                    state.critic_params, transitions.observation, pi
                )
                lmbda = jax.lax.stop_gradient(self._alpha / jnp.abs(q).mean())
                # actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
                policy_loss = -q.mean()
                bc_loss = jnp.mean(jnp.square(pi - transitions.action))
                actor_loss = lmbda * policy_loss + bc_loss
                metrics = {
                    "bc_loss": bc_loss,
                    "policy_loss": policy_loss,
                    "lambda": lmbda,
                    "actor_loss": actor_loss,
                }
                return actor_loss, metrics

            grad, metrics = jax.grad(loss_fn, has_aux=True)(state.policy_params)
            update, policy_opt_state = self._policy_optimizer.update(
                grad, state.policy_opt_state
            )
            policy_params = optax.apply_updates(state.policy_params, update)
            state = state._replace(
                policy_params=policy_params, policy_opt_state=policy_opt_state
            )
            return (state, metrics)

        @jax.jit
        def _update_critic(state: TrainingState, transitions: types.Transition):
            policy_key, key = jax.random.split(state.key)

            def loss_fn(critic_params):
                # Select action according to policy and add clipped noise
                noise = jnp.clip(
                    jax.random.normal(policy_key, transitions.action.shape)
                    * self._policy_noise,
                    -self._noise_clip,
                    self._noise_clip,
                )

                next_action = jnp.clip(
                    policy_network.apply(
                        state.policy_target_params, transitions.next_observation
                    )
                    + noise,
                    -1.0,
                    1.0,
                )

                # Compute the target Q value
                target_q1, target_q2 = critic_network.apply(
                    state.critic_target_params,
                    transitions.next_observation,
                    next_action,
                )
                target_q = jnp.minimum(target_q1, target_q2)
                target_q = jax.lax.stop_gradient(
                    transitions.reward
                    + transitions.discount * self._discount * target_q
                )

                # Get current Q estimates
                current_q1, current_q2 = critic_network.apply(
                    critic_params, transitions.observation, transitions.action
                )

                critic_loss = mse_loss(current_q1, target_q) + mse_loss(
                    current_q2, target_q
                )
                metrics = {
                    "q1": current_q1.mean(0),
                    "q2": current_q2.mean(0),
                    "critic_loss": critic_loss,
                }
                return critic_loss, metrics

            grad, metrics = jax.grad(loss_fn, has_aux=True)(state.critic_params)
            update, critic_opt_state = self._critic_optimizer.update(
                grad, state.critic_opt_state
            )
            critic_params = optax.apply_updates(state.critic_params, update)
            state = state._replace(
                critic_params=critic_params, critic_opt_state=critic_opt_state, key=key
            )
            return (state, metrics)

        @jax.jit
        def _update_target(state: TrainingState):
            return state._replace(
                policy_target_params=optax.incremental_update(
                    state.policy_params, state.policy_target_params, tau
                ),
                critic_target_params=optax.incremental_update(
                    state.critic_params,
                    state.critic_target_params,
                    tau,
                ),
            )

        def sgd_step(state: TrainingState, transitions: types.Transition, step):
            metrics = {}
            state, critic_metrics = _update_critic(state, transitions)
            metrics.update(critic_metrics)

            # Delayed policy updates
            if step % self._policy_update_period == 0:
                state, actor_metrics = _update_actor(state, transitions)
                metrics.update(actor_metrics)
                # Update target network parameters, notice that in the original TD3
                # formulation, the target is updated at the same time the policy is updated.
                state = _update_target(state)
            return state, metrics

        self._sgd_step = sgd_step

    def step(self):
        # Sample replay buffer
        transitions: types.Transition = next(self._data_iterator).data
        self._state, metrics = self._sgd_step(
            self._state, transitions, self._learning_steps
        )
        self._learning_steps += 1
        counts = self._counter.increment(steps=1)
        metrics.update(counts)
        self._logger.write(metrics)

    def get_variables(self, names: List[str]):
        variables = {
            "policy": self._state.policy_params,
            "critic": self._state.critic_params,
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        return self._state

    def load(self, state: TrainingState) -> None:
        self._state = state
