from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import jax.numpy as np

TASK_HORIZON = 150
NTRAIN_ITERS = 100
NROLLOUTS_PER_ITER = 1
PLAN_HOR = 25
MODEL_IN, MODEL_OUT = 27, 20


def get_goal(env):
  return env.ac_goal_pos


def obs_preproc(obs):
  return obs


def obs_postproc(obs, pred):
  return obs + pred


def targ_proc(obs, next_obs):
  return next_obs - obs


def obs_cost_fn(obs, goal):
  to_w, og_w = 0.5, 1.25
  tip_pos, obj_pos, goal_pos = obs[:, 14:17], obs[:, 17:20], goal

  tip_obj_dist = np.sum(np.abs(tip_pos - obj_pos), axis=1)
  obj_goal_dist = np.sum(np.abs(goal_pos - obj_pos), axis=1)
  return to_w * tip_obj_dist + og_w * obj_goal_dist


def ac_cost_fn(acs):
  return 0.1 * np.sum(np.square(acs), axis=1)


def cost_fn(obs, acs, goal):
  return obs_cost_fn(obs, goal) + ac_cost_fn(acs)
