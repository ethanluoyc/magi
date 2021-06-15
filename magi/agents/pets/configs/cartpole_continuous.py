from typing import Sequence, Callable

import dataclasses
import jax
from magi.environments import reward_fns
from magi.environments import termination_fns


@dataclasses.dataclass
class CartPoleContinuousConfig:
  task_horizon: int = 200
  hidden_sizes: Sequence[int] = (200, 200, 200)
  population_size: int = 500
  activation: Callable = jax.nn.silu
  planning_horizon: float = 15
  cem_alpha: float = 0.1
  cem_elite_frac: float = 0.1
  cem_return_mean_elites: bool = True
  weight_decay: float = 5e-5
  lr: float = 1e-3
  min_delta: float = 0.01
  num_ensembles: int = 5
  num_particles: int = 20
  num_epochs: int = 50
  patience: int = 50

  @staticmethod
  def obs_preproc(obs):
    # Consistent with mbrl to not use any transformation on inputs
    # return jnp.concatenate(
    #     [jnp.sin(obs[:, 1:2]),
    #      jnp.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)
    return obs

  @staticmethod
  def obs_postproc(obs, pred):
    return obs + pred

  @staticmethod
  def targ_proc(obs, next_obs):
    return next_obs - obs

  @staticmethod
  def cost_fn(x, a, goal):
    del goal
    return -reward_fns.cartpole(a, x)

  @staticmethod
  def termination_fn(x, a, goal):
    del goal
    return termination_fns.cartpole(a, x)
