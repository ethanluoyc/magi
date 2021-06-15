from typing import Union
from absl import logging
from acme import core
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tree

from magi.agents.pets import optimizers
from magi.agents.pets.dataset import ReplayBuffer

import tensorflow_probability as tfp

tfd = tfp.experimental.substrates.jax.distributions


def _make_trajectory_cost_fn(model, num_particles, cost_fn, terminal_fn):
  """Make stochastic cost function used by random optimizer or cem."""

  def cost(actions, key, params, state, x_init, goal):
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
    # output is [T, P * B, D]
    _, unrolled_states = model.unroll(params, state, key, xinit_particles, actions)
    # costs, dones is [T, P * B],
    # map across time horizon to get per-step costs
    # this should be more efficient than computing the cost in unroll
    # goal_cost_fn = functools.partial(cost_fn, goal=goal)
    # goal_terminal_fn = functools.partial(terminal_fn, goal=goal)
    costs = jax.vmap(cost_fn, in_axes=(0, 0, None))(unrolled_states, actions, goal)
    # import jax.experimental.host_callback as hcb
    # hcb.id_print(costs[:, 0], what="costs")
    dones = jax.vmap(terminal_fn, in_axes=(0, 0, None))(unrolled_states, actions, goal)
    costs, dones = tree.map_structure(
        lambda x: x.reshape((horizon, num_particles, batch_size)), (costs, dones))
    total_costs = jnp.zeros((
        num_particles,
        batch_size,
    ))
    terminated = jnp.zeros((num_particles, batch_size), dtype=jnp.bool_)
    for t in range(horizon):
      c_t = costs[t]
      c_t = jnp.where(terminated, 0, c_t)
      # print(c_t)
      terminated = jnp.logical_or(dones[t], terminated)
      total_costs = total_costs + c_t
    return jnp.mean(total_costs, axis=0)

  return cost


class OptimizerBasedActor(core.Actor):

  def __init__(
      self,
      spec,
      model,
      cost_fn,
      terminal_fn,
      dataset: ReplayBuffer,
      variable_client,
      controller_fn,
      time_horizon: int,
      num_particles: int = 20,
      seed: Union[jnp.ndarray, int] = 1,
      num_initial_episodes: int = 1,
  ):
    self._spec = spec
    self._controller_fn = controller_fn
    self._time_horizon = time_horizon
    self._rng = hk.PRNGSequence(seed)
    self._last_timestep = None
    self._first_trial = True
    self._num_initial_episodes = num_initial_episodes
    self._num_episodes_seen = 0
    self._goal = None

    self.cost_fn = _make_trajectory_cost_fn(model, num_particles, cost_fn, terminal_fn)
    self._client = variable_client
    self.dataset = dataset
    self._extras = None

    action_spec = spec.actions
    action_shape = (time_horizon,) + action_spec.shape
    lower_bound = np.broadcast_to(action_spec.minimum, action_shape)
    upper_bound = np.broadcast_to(action_spec.maximum, action_shape)

    self._initial_solution = np.ones(shape=(time_horizon, *action_spec.shape)) * (
        (lower_bound + upper_bound) / 2)
    self._last_actions = None

  def update_goal(self, goal):
    self._goal = goal

  def select_action(self, observation: np.ndarray):
    # [T, A]
    if self._num_episodes_seen < self._num_initial_episodes:
      lb = np.broadcast_to(self._spec.actions.minimum, self._spec.actions.shape)
      ub = np.broadcast_to(self._spec.actions.maximum, self._spec.actions.shape)
      action = np.asarray((tfd.Uniform(low=lb, high=ub).sample(seed=next(self._rng))))
      assert action.shape == self._spec.actions.shape
      return action
    variables = self._client.params
    params = variables['params']
    normalizer = variables['state']
    key = next(self._rng)
    actions, extras = self._controller_fn(self.cost_fn, params, normalizer, observation,
                                          self._last_actions, key, self._goal)
    self._extras = extras
    self._last_actions = np.asarray(actions)
    return np.array(actions[0])

  def observe_first(self, timestep: dm_env.TimeStep):
    self._last_timestep = timestep
    self._last_actions = self._initial_solution

  def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep):
    # Add a transition to the dataset
    self.dataset.add(self._last_timestep.observation, action, next_timestep.observation,
                     next_timestep.reward, next_timestep.last())
    # Update the last observation
    self._last_timestep = next_timestep
    # TODO use initial solution
    self._last_actions = np.roll(self._last_actions, -1)
    self._last_actions[:-1] = self._initial_solution[0]
    if next_timestep.last():
      logging.info('Final planning cost %.3f', self._extras)
      self._num_episodes_seen += 1

  def update(self, wait=True):
    self._client.update_and_wait()


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
      num_initial_episodes=1,
  ):
    self.dataset = dataset
    self.time_horizon = time_horizon

    def controller(cost_fn, params, state, observation, initial_solution, key, goal):
      action_spec = spec.actions
      action_shape = (time_horizon,) + action_spec.shape
      lower_bound = np.broadcast_to(action_spec.minimum, action_shape)
      upper_bound = np.broadcast_to(action_spec.maximum, action_shape)
      return optimizers.minimize_cem(cost_fn,
                                     initial_solution,
                                     key,
                                     args=(params, state, observation, goal),
                                     bounds=(lower_bound, upper_bound),
                                     n_iterations=n_iterations,
                                     population_size=pop_size,
                                     elite_fraction=elite_frac,
                                     alpha=alpha,
                                     fn_use_key=True,
                                     return_mean_elites=return_mean_elites)

    controller = jax.jit(controller, static_argnums=0)
    super().__init__(spec, forward_fn, cost_fn, terminal_fn, dataset, variable_client,
                     controller, time_horizon, num_particles, seed,
                     num_initial_episodes)


class RandomOptimizerActor(OptimizerBasedActor):

  def __init__(
      self,
      spec,
      network,
      cost_fn,
      terminal_fn,
      dataset,
      variable_client,
      num_samples=2000,
      time_horizon=25,
      num_particles=20,
      seed=0,
      num_initial_episodes=1,
  ):

    def controller(cost_fn, params, state, observation, initial_solution, key, goal):
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

    super().__init__(spec, network, cost_fn, terminal_fn, dataset, variable_client,
                     controller, time_horizon, num_particles, seed,
                     num_initial_episodes)
