"""Implementations of Implicit Q Learning (IQL) learner component."""
import time
from typing import Any, Dict, Iterator, NamedTuple, Tuple

import acme
from acme import types as core_types
from acme.jax import networks as networks_lib
from acme.jax import types
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax

from magi.agents.iql import networks as iql_networks

_Metrics = Dict[str, jnp.ndarray]


class TrainingState(NamedTuple):
    policy_params: networks_lib.Params
    policy_opt_state: optax.OptState
    value_params: networks_lib.Params
    value_opt_state: optax.OptState
    critic_params: networks_lib.Params
    critic_opt_state: optax.OptState
    target_critic_params: networks_lib.Params
    steps: int


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


class IQLLearner(acme.Learner):
    """IQL Learner."""

    _state: TrainingState

    def __init__(
        self,
        random_key: types.PRNGKey,
        networks: iql_networks.IQLNetworks,
        dataset: Iterator[core_types.Transition],
        policy_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        value_optimizer: optax.GradientTransformation,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        counter=None,
        logger=None,
    ):
        """Create an instance of the IQLLearner.

        Args:
            random_key (types.PRNGKey): random seed used by the learner.
            networks (iql_networks.IQLNetworks): networks used by the learner.
            dataset (Iterator[core_types.Transition]): dataset iterator.
            policy_optimizer (optax.GradientTransformation): optimizer for policy.
            critic_optimizer (optax.GradientTransformation): optimizer for critic.
            value_optimizer (optax.GradientTransformation): optimizer for value critic.
            discount (float, optional): additional discount. Defaults to 0.99.
            tau (float, optional): target soft update rate. Defaults to 0.005.
            expectile (float, optional): expectile for training critic. Defaults to 0.8.
            temperature (float, optional): temperature for the AWR. Defaults to 0.1.
            counter ([type], optional): counter for keeping counts. Defaults to None.
            logger ([type], optional): logger for writing metrics. Defaults to None.

        Returns:
            An instance of IQLLearner
        """

        policy_network = networks.policy_network
        value_network = networks.value_network
        critic_network = networks.critic_network

        def make_initial_state(key):
            policy_key, critic_key, value_key = jax.random.split(key, 3)
            policy_params = policy_network.init(policy_key)
            policy_opt_state = policy_optimizer.init(policy_params)
            critic_params = critic_network.init(critic_key)
            critic_opt_state = critic_optimizer.init(critic_params)
            value_params = value_network.init(value_key)
            value_opt_state = value_optimizer.init(value_params)
            state = TrainingState(
                policy_params=policy_params,
                policy_opt_state=policy_opt_state,
                critic_params=critic_params,
                critic_opt_state=critic_opt_state,
                target_critic_params=critic_params,
                value_params=value_params,
                value_opt_state=value_opt_state,
                steps=0,
            )
            return state

        def awr_update_actor(
            policy_params: networks_lib.Params,
            policy_opt_state: optax.OptState,
            target_critic_params: networks_lib.Params,
            value_params: networks_lib.Params,
            batch: core_types.Transition,
        ):
            v = value_network.apply(value_params, batch.observation)
            q1, q2 = critic_network.apply(
                target_critic_params, batch.observation, batch.action
            )
            q = jnp.minimum(q1, q2)
            exp_a = jnp.exp((q - v) * temperature)
            exp_a = jnp.minimum(exp_a, 100.0)

            def actor_loss_fn(
                policy_params: networks_lib.Params,
            ) -> Tuple[jnp.ndarray, Any]:
                dist = policy_network.apply(policy_params, batch.observation)
                log_probs = dist.log_prob(batch.action)
                actor_loss = -(exp_a * log_probs).mean()

                return actor_loss, {"actor_loss": actor_loss, "adv": q - v}

            grads, info = jax.grad(actor_loss_fn, has_aux=True)(policy_params)
            updates, policy_opt_state = policy_optimizer.update(
                grads, policy_opt_state, policy_params
            )
            policy_params = optax.apply_updates(policy_params, updates)
            return policy_params, policy_opt_state, info

        def target_update(critic_params, target_critic_params):
            new_target_params = jax.tree_multimap(
                lambda p, tp: p * tau + tp * (1 - tau),
                critic_params,
                target_critic_params,
            )
            return new_target_params

        def update_v(
            value_params: networks_lib.Params,
            value_opt_state: optax.OptState,
            target_critic_params: networks_lib.Params,
            batch: core_types.Transition,
        ):
            q1, q2 = critic_network.apply(
                target_critic_params, batch.observation, batch.action
            )
            q = jnp.minimum(q1, q2)

            def value_loss_fn(
                value_params: networks_lib.Params,
            ) -> Tuple[jnp.ndarray, Any]:
                v = value_network.apply(value_params, batch.observation)
                value_loss = expectile_loss(q - v, expectile).mean()
                return value_loss, {
                    "value_loss": value_loss,
                    "v": v.mean(),
                }

            grads, info = jax.grad(value_loss_fn, has_aux=True)(value_params)
            updates, value_opt_state = value_optimizer.update(
                grads, value_opt_state, value_params
            )
            value_params = optax.apply_updates(value_params, updates)
            return value_params, value_opt_state, info

        def update_q(
            critic_params: networks_lib.Params,
            critic_opt_state: optax.OptState,
            target_value_params: networks_lib.Params,
            batch: core_types.Transition,
        ):
            next_v = value_network.apply(target_value_params, batch.next_observation)

            target_q = batch.reward + discount * batch.discount * next_v

            def critic_loss_fn(
                critic_params: networks_lib.Params,
            ) -> Tuple[jnp.ndarray, Any]:
                q1, q2 = critic_network.apply(
                    critic_params, batch.observation, batch.action
                )
                critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
                return critic_loss, {
                    "critic_loss": critic_loss,
                    "q1": q1.mean(),
                    "q2": q2.mean(),
                }

            grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic_params)
            updates, critic_opt_state = critic_optimizer.update(
                grads, critic_opt_state, critic_params
            )
            critic_params = optax.apply_updates(critic_params, updates)
            return critic_params, critic_opt_state, info

        def _update_step(
            state: TrainingState, batch: core_types.Transition
        ) -> Tuple[TrainingState, _Metrics]:
            value_params, value_opt_state, value_info = update_v(
                state.value_params,
                state.value_opt_state,
                state.target_critic_params,
                batch,
            )
            policy_params, policy_opt_state, actor_info = awr_update_actor(
                state.policy_params,
                state.policy_opt_state,
                state.target_critic_params,
                value_params,
                batch,
            )

            critic_params, critic_opt_state, critic_info = update_q(
                state.critic_params, state.critic_opt_state, value_params, batch
            )

            target_critic_params = target_update(
                critic_params, state.target_critic_params
            )
            state = TrainingState(
                policy_params=policy_params,
                policy_opt_state=policy_opt_state,
                critic_params=critic_params,
                critic_opt_state=critic_opt_state,
                value_params=value_params,
                value_opt_state=value_opt_state,
                target_critic_params=target_critic_params,
                steps=state.steps + 1,
            )
            return state, {**critic_info, **value_info, **actor_info}

        self._update_step = jax.jit(_update_step)
        self._state = make_initial_state(random_key)
        self._iterator = dataset
        self._logger = logger or loggers.make_default_logger("learner", save_data=False)
        self._counter = counter or counting.Counter()
        self._timestamp = None

    def step(self):
        # Get data from replay
        sample = next(self._iterator)
        transitions: types.Transition = sample.data
        # Perform a single learner step
        self._state, metrics = self._update_step(self._state, transitions)

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

    def save(self) -> TrainingState:
        return self._state
