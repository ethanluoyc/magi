from magi.environments import reward_fns
from magi.environments import termination_fns

TASK_HORIZON = 200

def obs_preproc(obs):
  return obs
  # return jnp.concatenate(
  #     [jnp.sin(obs[:, 1:2]),
  #      jnp.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)


def obs_postproc(obs, pred):
  return obs + pred


def targ_proc(obs, next_obs):
  return next_obs - obs


def cost_fn(x, a, goal):
  del goal
  return -reward_fns.cartpole(a, x)


def termination_fn(x, a, goal):
  del goal
  return termination_fns.cartpole(a, x)
