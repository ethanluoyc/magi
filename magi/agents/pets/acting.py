import functools

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from acme import core
from jax import lax

from magi.agents.pets import controllers


class OptimizerBasedActor(core.Actor):

  def __init__(
      self,
      spec,
      net_fn,
      cost_fn,
      dataset,
      variable_client,
      controller_fn,
      obs_preprocess,
      target_postprocess,
      seed=1,
      num_particles=20,
      num_ensembles=5,
  ):
    self._spec = spec
    self.num_updates = 0
    self._last_timestep = None
    self.net = hk.without_apply_rng(hk.transform(net_fn))
    self._rng = hk.PRNGSequence(seed)
    self._controller_fn = controller_fn
    self._num_particles = num_particles
    self._num_ensembles = num_ensembles
    self._obs_preprocess = obs_preprocess
    self._target_postprocess = target_postprocess

    def unroll(ensem_params, x_init_parts, actions, rng):
      # ensem_params [B, ...]
      # x_init_parts [P, dim_d]
      # actions [T, dim_a]
      # rng: random key
      # x_init_parts = self._obs_preprocess(x_init_parts)
      x_reshaped = jnp.reshape(
          x_init_parts,
          (num_ensembles, num_particles // num_ensembles, x_init_parts.shape[-1]))

      # [B, P // B, dim_d]
      def step(s, a):
        x, rng = s
        proc_x = jax.vmap(self._obs_preprocess, 0)(x)
        rng, rng_subkey = jax.random.split(rng)
        assert a.ndim == 1
        a_tiled = jnp.tile(jnp.expand_dims(a, 0), (
            proc_x.shape[1],
            1,
        ))
        mean, std = jax.vmap(self.net.apply, (0, 0, None))(ensem_params, proc_x,
                                                           a_tiled)
        # Note that we add x since we are predicting difference
        next_x = x + mean + jax.random.normal(
            rng_subkey, shape=std.shape, dtype=std.dtype) * std
        return (next_x, rng), next_x

      states = lax.scan(step, (x_reshaped, rng), actions)[1]
      states = jnp.reshape(states, (actions.shape[0], num_particles, states.shape[-1]))
      return states

    @jax.jit
    def model_cost_fn(ensem_params, rng, xinit, actions):
      xinit_particles = jnp.broadcast_to(xinit, (num_particles,) + xinit.shape)
      states = unroll(ensem_params, xinit_particles, actions, rng)  # (T, P, state)
      costs = jax.vmap(cost_fn, (1, None))(states, actions)  # (T, P)
      return jnp.sum(jnp.mean(costs, -1))

    self.cost_fn = model_cost_fn
    self._client = variable_client
    self.dataset = dataset

  def select_action(self, observation: np.ndarray):
    actions = self._controller_fn(
        self.cost_fn,
        self._client.params,
        observation,
        next(self._rng),
    )
    action = actions[0]  # MPC
    return np.array(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._last_timestep = timestep

  def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep):
    # Add a transition to the dataset
    self.dataset.add(
        self._last_timestep.observation,
        next_timestep.observation,
        action,
        next_timestep.reward,
    )
    # Update the last observation
    self._last_timestep = next_timestep

  def update(self):
    self._client.update()


class CEMOptimizerActor(OptimizerBasedActor):

  def __init__(
      self,
      spec,
      net_fn,
      cost_fn,
      dataset,
      variable_client,
      obs_preprocess,
      target_postprocess,
      time_horizon=25,
      n_iterations=5,
      pop_size=400,
      elite_frac=0.1,
      alpha=0.1,
      return_mean_elites=False,
      seed=0,
  ):
    self.dataset = dataset
    self.time_horizon = time_horizon

    def controller(cost_fn, params, observation, initial_solution, key):
      return controllers.cem_controller(cost_fn,
                                        spec.actions,
                                        params,
                                        observation,
                                        initial_solution,
                                        key,
                                        n_iterations=n_iterations,
                                        pop_size=pop_size,
                                        elite_frac=elite_frac,
                                        alpha=alpha,
                                        time_horizon=time_horizon,
                                        return_mean_elites=return_mean_elites)

    controller = jax.jit(controller, static_argnums=(0))
    # controller_fn = jax.jit(controller_fn, static_argnums=(0, 1))
    super().__init__(spec, net_fn, cost_fn, dataset, variable_client, controller,
                     obs_preprocess, target_postprocess, seed)
    action_spec = self._spec.actions
    action_shape = (self.time_horizon,) + action_spec.shape
    lower_bound = np.broadcast_to(action_spec.minimum, action_shape)
    upper_bound = np.broadcast_to(action_spec.maximum, action_shape)
    self._initial_solution = np.ones(shape=(self.time_horizon, *action_spec.shape)) * (
        (lower_bound + upper_bound) / 2)

  def select_action(self, observation: np.ndarray):
    actions = self._controller_fn(
        self.cost_fn,
        self._client.params,
        observation,
        self._last_actions,
        next(self._rng),
    )
    self._last_actions = np.asarray(actions)
    return np.array(actions[0])

  def observe_first(self, timestep: dm_env.TimeStep):
    self._last_timestep = timestep
    self._last_actions = self._initial_solution

  def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep):
    # Add a transition to the dataset
    self.dataset.add(
        self._last_timestep.observation,
        next_timestep.observation,
        action,
        next_timestep.reward,
    )
    # Update the last observation
    self._last_timestep = next_timestep
    # TODO use initial solution
    self._last_actions = np.roll(self._last_actions, -1)
    self._last_actions[:-1] = self._initial_solution[0]

  def update(self):
    self._client.update()


class RandomOptimizerActor(OptimizerBasedActor):

  def __init__(
      self,
      spec,
      net_fn,
      cost_fn,
      dataset,
      variable_client,
      obs_preprocess,
      target_postprocess,
      num_samples=2000,
      time_horizon=25,
      seed=0,
  ):
    self.num_samples = num_samples
    self.time_horizon = time_horizon

    def controller(cost_fn, params, observation, key):
      return controllers.random_controller(
          cost_fn,
          spec.actions,
          params,
          observation,
          key,
          num_samples=num_samples,
          time_steps=time_horizon,
      )

    controller = jax.jit(controller, static_argnums=0)

    # controller_fn = jax.jit(controller_fn, static_argnums=(0, 1))
    super().__init__(spec, net_fn, cost_fn, dataset, variable_client, controller,
                     obs_preprocess, target_postprocess, seed)
