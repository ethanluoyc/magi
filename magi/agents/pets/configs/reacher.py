from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import jax.numpy as np
import jax
# import numpy as np

# class ReacherConfigModule:
#   # ENV_NAME = "MBRLReacher3D-v0"
TASK_HORIZON = 150
NTRAIN_ITERS = 100
NROLLOUTS_PER_ITER = 1
PLAN_HOR = 25
# MODEL_IN, MODEL_OUT = 24, 17


def get_goal(env):
  return env.goal


def obs_cost_fn(obs, goal):
  ee_pos = get_ee_pos(obs)
  assert len(obs.shape) == 2
  assert len(ee_pos.shape) == 2
  assert len(goal.shape) == 1
  return np.sum(np.square(ee_pos - goal), axis=1)


def ac_cost_fn(acs):
  return 0.01 * np.sum(np.square(acs), axis=1)


def cost_fn(obs, act, goal):
  return obs_cost_fn(obs, goal) + ac_cost_fn(act)


def get_ee_pos(states):
  theta1, theta2, theta3, theta4, theta5, theta6, theta7 = (
      states[:, :1],
      states[:, 1:2],
      states[:, 2:3],
      states[:, 3:4],
      states[:, 4:5],
      states[:, 5:6],
      states[:, 6:],
  )
  rot_axis = np.concatenate([
      np.cos(theta2) * np.cos(theta1),
      np.cos(theta2) * np.sin(theta1), -np.sin(theta2)
  ],
                            axis=1)
  rot_perp_axis = np.concatenate(
      [-np.sin(theta1), np.cos(theta1),
       np.zeros(theta1.shape)], axis=1)
  cur_end = np.concatenate([
      0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2), 0.1 * np.sin(theta1)
      + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188, -0.4 * np.sin(theta2)
  ],
                           axis=1)

  for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
    perp_all_axis = np.cross(rot_axis, rot_perp_axis)
    x = np.cos(hinge) * rot_axis
    y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
    z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
    new_rot_axis = x + y + z
    new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
    # new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = \
    #     rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
    # Change to this
    # TODO: (check if this is correct)
    new_rot_perp_axis = np.where(
        np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True) < 1e-8, rot_perp_axis,
        new_rot_perp_axis)

    new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
    rot_axis, rot_perp_axis, cur_end = (new_rot_axis, new_rot_perp_axis,
                                        cur_end + length * new_rot_axis)
  return cur_end


def obs_preproc(obs):
  return obs


def obs_postproc(obs, pred):
  return obs + pred


def targ_proc(obs, next_obs):
  return next_obs - obs
