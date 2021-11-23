"""Implementations of Implicit Q Learning (IQL) learner component."""
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from magi.agents.iql import networks as networks_lib
from magi.agents.iql.types import Batch
from magi.agents.iql.types import InfoDict
from magi.agents.iql.types import Params
from magi.agents.iql.types import PRNGKey


class TrainingState(NamedTuple):
    policy_params: Params
    policy_opt_state: optax.OptState
    value_params: Params
    value_opt_state: optax.OptState
    critic_params: Params
    critic_opt_state: optax.OptState
    target_critic_params: Params


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


class Learner:

    _state: TrainingState
    rng: PRNGKey

    def __init__(
        self,
        seed: int,
        networks: networks_lib.IQLNetworks,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        max_steps: Optional[int] = None,
        opt_decay_schedule: str = "cosine",
    ):

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            policy_optimizer = optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(schedule_fn),
            )
        else:
            policy_optimizer = optax.adam(learning_rate=actor_lr)
        value_optimizer = optax.adam(learning_rate=value_lr)
        critic_optimizer = optax.adam(learning_rate=critic_lr)

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
            )
            return state

        def awr_update_actor(
            key: PRNGKey,
            policy_params: Params,
            policy_opt_state: optax.OptState,
            target_critic_params: Params,
            value_params: Params,
            batch: Batch,
        ):
            v = value_network.apply(value_params, batch.observations)
            q1, q2 = critic_network.apply(
                target_critic_params, batch.observations, batch.actions
            )
            q = jnp.minimum(q1, q2)
            exp_a = jnp.exp((q - v) * temperature)
            exp_a = jnp.minimum(exp_a, 100.0)

            def actor_loss_fn(policy_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
                dist = policy_network.apply(
                    policy_params,
                    batch.observations,
                    training=True,
                    rngs={"dropout": key},
                )
                log_probs = dist.log_prob(batch.actions)
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
            value_params: Params,
            value_opt_state: Params,
            target_critic_params: Params,
            batch: Batch,
        ):
            actions = batch.actions
            q1, q2 = critic_network.apply(
                target_critic_params, batch.observations, actions
            )
            q = jnp.minimum(q1, q2)

            def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
                v = value_network.apply(value_params, batch.observations)
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
            critic_params: Params,
            critic_opt_state: optax.OptState,
            target_value_params: Params,
            batch: Batch,
        ):
            next_v = value_network.apply(target_value_params, batch.next_observations)

            target_q = batch.rewards + discount * batch.masks * next_v

            def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
                q1, q2 = critic_network.apply(
                    critic_params, batch.observations, batch.actions
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
            rng: PRNGKey, state: TrainingState, batch: Batch
        ) -> Tuple[PRNGKey, TrainingState, InfoDict]:
            value_params, value_opt_state, value_info = update_v(
                state.value_params,
                state.value_opt_state,
                state.target_critic_params,
                batch,
            )
            key, rng = jax.random.split(rng)
            policy_params, policy_opt_state, actor_info = awr_update_actor(
                key,
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
            )
            return rng, state, {**critic_info, **value_info, **actor_info}

        self._update_step = jax.jit(_update_step)
        random_key, init_key = jax.random.split(jax.random.PRNGKey(seed))
        self._state = make_initial_state(init_key)
        self.rng = random_key

    def update(self, batch: Batch) -> InfoDict:
        self.rng, self._state, info = self._update_step(self.rng, self._state, batch)
        return info

    def get_variables(self, names):
        variables = {
            "policy": self._state.policy_params,
        }
        return [variables[name] for name in names]

    def restore(self, state: TrainingState):
        self._state = state

    def save(self) -> TrainingState:
        return self._state
