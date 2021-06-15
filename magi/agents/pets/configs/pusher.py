from typing import Callable

import dataclasses

import jax.numpy as np
from magi.agents.pets.configs.default import Config
import jax


@dataclasses.dataclass
class PusherConfig(Config):
  task_horizon: int = 150
  time_horizon: int = 25
  activation: Callable = jax.nn.swish

  @staticmethod
  def get_goal(env):
    return env.ac_goal_pos

  @staticmethod
  def obs_preproc(obs):
    return obs

  @staticmethod
  def obs_postproc(obs, pred):
    return obs + pred

  @staticmethod
  def targ_proc(obs, next_obs):
    return next_obs - obs

  @staticmethod
  def obs_cost_fn(obs, goal):
    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos, goal_pos = obs[:, 14:17], obs[:, 17:20], goal

    tip_obj_dist = np.sum(np.abs(tip_pos - obj_pos), axis=1)
    obj_goal_dist = np.sum(np.abs(goal_pos - obj_pos), axis=1)
    return to_w * tip_obj_dist + og_w * obj_goal_dist

  @staticmethod
  def ac_cost_fn(acs):
    return 0.1 * np.sum(np.square(acs), axis=1)

  @staticmethod
  def cost_fn(obs, acs, goal):
    return PusherConfig.obs_cost_fn(obs, goal) + PusherConfig.ac_cost_fn(acs)
