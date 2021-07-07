from typing import Union

from absl import logging
from acme import core
from acme import specs
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability as tfp

from magi.agents.pets import models
from magi.agents.pets import optimizers
from magi.agents.pets import replay as replay_lib

tfd = tfp.experimental.substrates.jax.distributions


def _make_trajectory_cost_fn(model, num_particles: int):
    """Make stochastic cost function used by random optimizer or cem."""

    def cost(actions, key, params, state, x_init, goal):
        """Objective function for trajectory planning.
        Args:
          actions: [B, T, A]
          x_init: [D]
          params, key, num_particles
        """
        rewards = model.unroll(params, state, key, x_init, actions, goal, num_particles)
        return -rewards

    return cost


class OptimizerBasedActor(core.Actor):
    def __init__(
        self,
        spec: specs.EnvironmentSpec,
        model_env: models.ModelEnv,
        replay: replay_lib.ReplayBuffer,
        variable_client: core.VariableSource,
        controller_fn,
        planning_horizon: int,
        num_particles: int = 20,
        seed: Union[jnp.ndarray, int] = 1,
        num_initial_episodes: int = 1,
    ):
        self._spec = spec
        self._replay = replay
        self._planning_horizon = planning_horizon
        self._rng = hk.PRNGSequence(seed)
        self._controller_fn = controller_fn
        self._num_initial_episodes = num_initial_episodes
        self._num_episodes_seen = 0

        self._cost_fn = _make_trajectory_cost_fn(model_env, num_particles)
        self._client = variable_client
        self._extras = None
        self._last_timestep = None
        self._first_trial = True
        self._goal = None

        action_spec = spec.actions
        action_shape = (planning_horizon,) + action_spec.shape
        lower_bound = np.broadcast_to(action_spec.minimum, action_shape)
        upper_bound = np.broadcast_to(action_spec.maximum, action_shape)

        self._initial_solution = np.ones(
            shape=(planning_horizon, *action_spec.shape)
        ) * ((lower_bound + upper_bound) / 2)
        self._last_actions = None

    def update_goal(self, goal):
        self._goal = goal

    def select_action(self, observation: np.ndarray):
        # [T, A]
        if self._num_episodes_seen < self._num_initial_episodes:
            lb = np.broadcast_to(self._spec.actions.minimum, self._spec.actions.shape)
            ub = np.broadcast_to(self._spec.actions.maximum, self._spec.actions.shape)
            action = np.asarray(
                (tfd.Uniform(low=lb, high=ub).sample(seed=next(self._rng)))
            )
            assert action.shape == self._spec.actions.shape
            return action
        variables = self._client.params
        params = variables["params"]
        state = variables["state"]
        key = next(self._rng)
        actions, extras = self._controller_fn(
            self._cost_fn,
            params,
            state,
            observation,
            self._last_actions,
            key,
            self._goal,
        )
        self._extras = extras
        self._last_actions = np.asarray(actions)
        return np.array(actions[0])

    def observe_first(self, timestep: dm_env.TimeStep):
        self._last_timestep = timestep
        self._last_actions = self._initial_solution.copy()

    def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep):
        # Add a transition to the dataset
        self._replay.add(
            self._last_timestep.observation,
            action,
            next_timestep.observation,
            next_timestep.reward,
            next_timestep.last(),
        )
        # Update the last observation
        # TODO(yl): Move MPC trajectory optimization into a separate class
        # so to not clutter the actor.
        self._last_timestep = next_timestep
        self._last_actions = np.roll(self._last_actions, -1, axis=0)
        self._last_actions[-1:] = self._initial_solution[0].copy()
        if next_timestep.last():
            logging.info("Final planning cost %s", self._extras)
            self._num_episodes_seen += 1

    def update(self, wait=True):
        del wait
        self._client.update_and_wait()


class CEMOptimizerActor(OptimizerBasedActor):
    def __init__(
        self,
        spec: specs.EnvironmentSpec,
        model_env: models.ModelEnv,
        replay: replay_lib.ReplayBuffer,
        variable_client: core.VariableSource,
        planning_horizon: int = 25,
        n_iterations: int = 5,
        population_size: int = 400,
        elite_frac: float = 0.1,
        alpha: float = 0.1,
        return_mean_elites: bool = False,
        num_particles: int = 20,
        seed: int = 0,
        num_initial_episodes: int = 1,
    ):
        def controller(
            cost_fn, params, state, observation, initial_solution, key, goal
        ):
            action_spec = spec.actions
            action_shape = (planning_horizon,) + action_spec.shape
            lower_bound = np.broadcast_to(action_spec.minimum, action_shape)
            upper_bound = np.broadcast_to(action_spec.maximum, action_shape)
            return optimizers.minimize_cem(
                cost_fn,
                initial_solution,
                key,
                args=(params, state, observation, goal),
                bounds=(lower_bound, upper_bound),
                n_iterations=n_iterations,
                population_size=population_size,
                elite_fraction=elite_frac,
                alpha=alpha,
                fn_use_key=True,
                return_mean_elites=return_mean_elites,
            )

        controller = jax.jit(controller, static_argnums=0)
        super().__init__(
            spec,
            model_env,
            replay,
            variable_client,
            controller,
            planning_horizon,
            num_particles,
            seed,
            num_initial_episodes,
        )


class RandomOptimizerActor(OptimizerBasedActor):
    def __init__(
        self,
        spec,
        model_env,
        replay: replay_lib.ReplayBuffer,
        variable_client,
        num_samples: int = 2000,
        planning_horizon: int = 25,
        num_particles: int = 20,
        seed: int = 0,
        num_initial_episodes=1,
    ):
        def controller(
            cost_fn, params, state, observation, initial_solution, key, goal
        ):
            action_spec = spec.actions
            return optimizers.minimize_random(
                cost_fn,
                initial_solution,
                key,
                args=(params, state, observation, goal),
                bounds=(action_spec.minimum, action_spec.maximum),
                population_size=num_samples,
                fn_use_key=True,
            )

        controller = jax.jit(controller, static_argnums=0)

        super().__init__(
            spec,
            model_env,
            replay,
            variable_client,
            controller,
            planning_horizon,
            num_particles,
            seed,
            num_initial_episodes,
        )
