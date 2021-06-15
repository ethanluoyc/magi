from typing import Callable, Sequence
import dataclasses
import jax
from magi.environments import termination_fns


@dataclasses.dataclass
class Config:
  task_horizon: int = 1000
  hidden_sizes: Sequence[int] = (200, 200, 200)
  population_size: int = 500
  activation: Callable = jax.nn.silu
  planning_horizon: int = 15
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
    raise NotImplementedError()

  @staticmethod
  def obs_postproc(obs, pred):
    raise NotImplementedError()

  @staticmethod
  def targ_proc(obs, next_obs):
    raise NotImplementedError()

  @staticmethod
  def cost_fn(obs, acs, goal):
    raise NotImplementedError()

  @staticmethod
  def termination_fn(obs, act, goal):
    del goal
    return termination_fns.no_termination(act, obs)
