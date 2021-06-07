import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from acme import core
from jax import lax
import tree

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
      obs_postproc,
      seed=1,
      num_particles=20,
      num_ensembles=5,
  ):
    self._spec = spec
    self.num_updates = 0
    self._last_timestep = None
    self._rng = hk.PRNGSequence(seed)
    self._controller_fn = controller_fn
    self._num_particles = num_particles
    self._num_ensembles = num_ensembles
    self._obs_preprocess = obs_preprocess
    self._obs_postprocess = obs_postproc

    transformed = hk.without_apply_rng(hk.transform(net_fn))

    def forward_fn(ensem_params, key, observations, actions):
      states = self._obs_preprocess(observations)
      # mean, std = network_transformed.apply(ensem_params, state, action)
      # next_state =
      num_ensembles = tree.flatten(ensem_params)[0].shape[0]
      batch_size = states.shape[0]
      new_batch_size, ragged = divmod(batch_size, num_ensembles)
      if ragged:
        raise NotImplementedError(
            f"Ragged batch not supported. ({batch_size} % {num_ensembles} == {ragged})")
      reshaped_states, reshaped_act = tree.map_structure(
          lambda x: x.reshape((num_ensembles, new_batch_size, *x.shape[1:])),
          (states, actions))
      mean, std = jax.vmap(transformed.apply)(ensem_params, reshaped_states,
                                              reshaped_act)
      mean, std = tree.map_structure(lambda x: x.reshape((-1, *x.shape[2:])),
                                     (mean, std))
      output = jax.random.normal(key, shape=mean.shape) * std + mean
      return self._obs_postprocess(observations, output)

    def unroll(forward_fn, params, rng, x_init, actions):
      """Unroll model along a sequence of actions.
      Args:
        ensem_params: hk.Params.
        rng: JAX random key.
        x_init [B, D]
        actions [T, B, A]
      """

      def step(input_, a_t):
        rng, x_t = input_
        rng, rng_step = jax.random.split(rng)
        # import pdb; pdb.set_trace()
        x_tp1 = forward_fn(params, rng_step, x_t, a_t)
        return (rng, x_tp1), x_tp1

      return lax.scan(step, (rng, x_init), actions)

    def model_cost_fn(params, key, x_init, actions):
      # def objective(actions, x_init, params, key):
      """Objective function for trajectory planning.
      Args:
        actions: [P, T, A]
        x_init: [D]
        params, key, num_particles
      """
      num_particles = tree.flatten(actions)[0].shape[0]
      # xinit_particles has shape [P, D]
      xinit_particles = jnp.broadcast_to(x_init, (num_particles,) + x_init.shape)
      # actions now have shape [T, P, A]
      actions = jnp.swapaxes(actions, 0, 1)
      # unrolled_states has shape [T, P, D]
      _, unrolled_states = unroll(forward_fn, params, key, xinit_particles, actions)
      # costs is [T, P]
      # print(unrolled_states.shape)
      costs = jax.vmap(cost_fn)(unrolled_states, actions)
      # import pdb; pdb.set_trace()
      return jnp.sum(costs, axis=0)

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
      obs_postproc,
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
                     obs_preprocess, obs_postproc, seed)
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
      obs_postproces,
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
                     obs_preprocess, obs_postproces, seed)
