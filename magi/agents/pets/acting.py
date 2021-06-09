from acme import core
import dm_env
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import tree

from magi.agents.pets import optimizers


def _unroll(init_fn, forward_fn, params, rng, x_init, actions):
  """Unroll model along a sequence of actions.
  Args:
    ensem_params: hk.Params.
    rng: JAX random key.
    x_init [B, D]
    actions [T, B, A]
  """
  rng, rng_init = jax.random.split(rng)
  state = init_fn(params, rng_init, x_init)

  def step(input_, a_t):
    rng, x_t, state = input_
    rng, rng_step = jax.random.split(rng)
    x_tp1 = forward_fn(params, state, rng_step, x_t, a_t)
    return (rng, x_tp1, state), x_tp1

  return lax.scan(step, (rng, x_init, state), actions)


class OptimizerBasedActor(core.Actor):

  def __init__(
      self,
      spec,
      forward_fn,
      cost_fn,
      terminal_fn,
      dataset,
      variable_client,
      controller_fn,
      time_horizon: int,
      num_particles: int = 20,
      seed: int = 1,
  ):
    self._spec = spec
    self._controller_fn = controller_fn
    self._time_horizon = time_horizon
    self._rng = hk.PRNGSequence(seed)
    self._last_timestep = None

    init_fn, forward_fn = forward_fn

    def model_cost_fn(actions, key, params, x_init):
      """Objective function for trajectory planning.
      Args:
        actions: [B, T, A]
        x_init: [D]
        params, key, num_particles
      """
      batch_size = actions.shape[0]
      horizon = actions.shape[1]
      # [P, B, D]
      xinit_particles = jnp.broadcast_to(x_init,
                                         (num_particles, batch_size) + x_init.shape)
      # [P * B, D]
      xinit_particles = jnp.reshape(xinit_particles, (num_particles * batch_size, -1))
      # [P, B, T, A]
      actions = jnp.broadcast_to(actions, (num_particles,) + actions.shape)
      # [P * B, T, A]
      actions = jnp.reshape(actions, (num_particles * batch_size, horizon, -1))
      # actions now have shape [T, P * B, A]
      actions = jnp.swapaxes(actions, 0, 1)
      # unrolled_states has shape [T, P * B, D]
      # TODO(yl): Consider extracting this to a separate class
      _, unrolled_states = _unroll(init_fn, forward_fn, params, key, xinit_particles,
                                   actions)
      # costs, dones is [T, P * B], map across time horizon
      costs = jax.vmap(cost_fn)(unrolled_states, actions)
      dones = jax.vmap(terminal_fn)(unrolled_states, actions)
      costs, dones = tree.map_structure(
          lambda x: x.reshape((horizon, num_particles, batch_size)), (costs, dones))
      total_costs = jnp.zeros((
          num_particles,
          batch_size,
      ))
      terminated = jnp.zeros((num_particles, batch_size), dtype=jnp.bool_)
      for t in range(horizon):
        c_t = jnp.where(terminated, 0, costs[t])
        terminated = jnp.logical_or(dones[t], terminated)
        total_costs = total_costs + c_t
      return jnp.mean(total_costs, axis=0)

    self.cost_fn = model_cost_fn
    self._client = variable_client
    self.dataset = dataset

    action_spec = spec.actions
    action_shape = (time_horizon,) + action_spec.shape
    lower_bound = np.broadcast_to(action_spec.minimum, action_shape)
    upper_bound = np.broadcast_to(action_spec.maximum, action_shape)

    self._initial_solution = np.ones(shape=(time_horizon, *action_spec.shape)) * (
        (lower_bound + upper_bound) / 2)
    self._last_actions = None

  def select_action(self, observation: np.ndarray):
    # [T, A]
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

  def update(self, wait=True):
    self._client.update(wait)


class CEMOptimizerActor(OptimizerBasedActor):

  def __init__(
      self,
      spec,
      forward_fn,
      cost_fn,
      terminal_fn,
      dataset,
      variable_client,
      time_horizon=25,
      n_iterations=5,
      pop_size=400,
      elite_frac=0.1,
      alpha=0.1,
      return_mean_elites=False,
      num_particles=20,
      seed=0,
  ):
    self.dataset = dataset
    self.time_horizon = time_horizon

    def controller(cost_fn, params, observation, initial_solution, key):
      action_spec = spec.actions
      action_shape = (self.time_horizon,) + action_spec.shape
      lower_bound = np.broadcast_to(action_spec.minimum, action_shape)
      upper_bound = np.broadcast_to(action_spec.maximum, action_shape)
      return optimizers.minimize_cem(cost_fn,
                                     initial_solution,
                                     key,
                                     args=(params, observation),
                                     bounds=(lower_bound, upper_bound),
                                     n_iterations=n_iterations,
                                     population_size=pop_size,
                                     elite_fraction=elite_frac,
                                     alpha=alpha,
                                     fn_use_key=True,
                                     return_mean_elites=return_mean_elites)

    controller = jax.jit(controller, static_argnums=0)
    super().__init__(spec, forward_fn, cost_fn, terminal_fn, dataset, variable_client,
                     controller, time_horizon, num_particles, seed)


class RandomOptimizerActor(OptimizerBasedActor):

  def __init__(self,
               spec,
               network,
               cost_fn,
               terminal_fn,
               dataset,
               variable_client,
               num_samples=2000,
               time_horizon=25,
               num_particles=20,
               seed=0):

    def controller(cost_fn, params, observation, initial_solution, key):
      action_spec = spec.actions
      return optimizers.minimize_random(
          cost_fn,
          initial_solution,
          key,
          args=(params, observation),
          bounds=(action_spec.minimum, action_spec.maximum),
          population_size=num_samples,
          fn_use_key=True,
      )

    controller = jax.jit(controller, static_argnums=0)

    super().__init__(spec, network, cost_fn, terminal_fn, dataset, variable_client,
                     controller, time_horizon, num_particles, seed)
