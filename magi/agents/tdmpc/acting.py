from typing import Optional

import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme import adders
from acme import core
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import variable_utils

from magi.agents.tdmpc import networks as tdmpc_networks
from magi.agents.tdmpc import planning


class TDMPCActor(core.Actor):
    def __init__(
        self,
        variable_client: variable_utils.VariableClient,
        spec: specs.EnvironmentSpec,
        networks: tdmpc_networks.TDMPCNetworks,
        random_key: jax.random.PRNGKeyArray,
        *,
        std_schedule: optax.Schedule,
        horizon_schedule: optax.Schedule,
        discount: float = 0.99,
        num_samples: int = 512,
        min_std: float = 0.05,
        temperature: float = 0.5,
        momentum: float = 0.1,
        num_elites: int = 64,
        iterations: int = 6,
        seed_steps: int = 5000,
        mixture_coef: float = 0.05,
        horizon: int = 5,
        adder: Optional[adders.Adder] = None,
        evaluation: bool = False,
    ) -> None:
        action_dim = spec.actions.shape[0]
        num_pi_trajs = int(mixture_coef * num_samples)

        def random_policy(params, obs, state):
            del params, obs
            (key, prev_mean, epsilon) = state
            sample_key, key = jax.random.split(key)
            action = jax.random.uniform(
                key=sample_key,
                minval=-1.0,
                maxval=1.0,
                dtype=jnp.float32,
                shape=spec.actions.shape,
            )
            return action, (key, prev_mean, epsilon)

        self._random_policy = jax.jit(random_policy)

        def plan_init(key: jax.random.PRNGKeyArray, step: int):
            epsilon = std_schedule(step) if not evaluation else min_std
            planning_horizon = int(min(horizon, horizon_schedule(step)))
            previous_trajectory = planning.get_initial_trajectory(
                action_dim,
                planning_horizon,
            )
            return (key, previous_trajectory, epsilon)

        def plan_step(params, obs, state):
            encoder_fn = networks.h

            def policy_fn(params, z, key):
                return networks.pi(params, z, min_std, key)

            def critic_fn(params, z, action):
                return jnp.squeeze(jnp.minimum(*networks.q(params, z, action)), axis=-1)

            def model_fn(params, z, action):
                next_z, reward = networks.next(params, z, action)
                return next_z, jnp.squeeze(reward, axis=-1)

            (key, previous_trajectory, epsilon) = state

            step_key, sample_key, key = jax.random.split(key, 3)

            trajectory_mean, trajectory_std, _, actions = planning.td_mpc_planner(
                policy_fn=policy_fn,
                encoder_fn=encoder_fn,
                model_fn=model_fn,
                critic_fn=critic_fn,
                params=params,
                observation=obs,
                previous_trajectory=previous_trajectory,
                key=step_key,
                n_policy_trajectories=num_pi_trajs,
                n_sample_trajectories=num_samples,
                n_iterations=iterations,
                epsilon=epsilon,
                k=num_elites,
                discount=discount,
                temperature=temperature,
                momentum=momentum,
            )

            previous_trajectory = trajectory_mean
            mean, std = actions[0], trajectory_std[0]
            action = mean
            if not evaluation:
                action = action + std * jax.random.normal(sample_key, (action_dim,))
            new_state = (key, previous_trajectory, epsilon)
            return action, new_state

        self._plan_init = plan_init
        self._plan_step = jax.jit(plan_step)

        # TODO(yl): What to do with exploration in distribution experiments?
        self._seed_steps = seed_steps
        self._actor_steps = 0
        self._key = random_key

        self._state = None
        self._adder = adder
        self._variable_client = variable_client
        self._evaluation = evaluation

    @property
    def _params(self):
        return self._variable_client.params

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        init_key, self._key = jax.random.split(self._key)
        self._state = self._plan_init(init_key, self._actor_steps)

        if self._adder is not None:
            self._adder.add_first(timestep)

    def observe(
        self, action: networks_lib.Action, next_timestep: dm_env.TimeStep
    ) -> None:
        if self._adder is not None:
            self._adder.add(action, next_timestep)

    def select_action(
        self, observation: networks_lib.Observation
    ) -> networks_lib.Action:
        if self._actor_steps < self._seed_steps and not self._evaluation:
            action, self._state = self._random_policy(
                self._params, observation, self._state
            )
        else:
            action, self._state = self._plan_step(
                self._params, observation, self._state
            )

        self._actor_steps += 1
        return np.asarray(action)

    def update(self, wait: bool = False) -> None:
        self._variable_client.update(wait)
