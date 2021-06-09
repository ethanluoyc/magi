import jax.numpy as jnp

TIME_LIMIT = 200


def obs_preproc(obs):
  return jnp.concatenate(
      [jnp.sin(obs[:, 1:2]),
       jnp.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)


def obs_postproc(obs, pred):
  return obs + pred


def targ_proc(obs, next_obs):
  return next_obs - obs


def obs_cost_fn(obs):
  return -jnp.exp(
      -jnp.sum(jnp.square(_get_ee_pos(obs) - jnp.array([0.0, 0.6])), axis=1) / (0.6**2))


def ac_cost_fn(acs):
  return 0.01 * jnp.sum(jnp.square(acs), axis=1)


def cost_fn(obs, acs, goal):
  return obs_cost_fn(obs) + ac_cost_fn(acs)


def _get_ee_pos(obs):
  x0, theta = obs[:, :1], obs[:, 1:2]
  return jnp.concatenate([x0 - 0.6 * jnp.sin(theta), -0.6 * jnp.cos(theta)], axis=1)
