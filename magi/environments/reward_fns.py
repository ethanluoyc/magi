import jax.numpy as jnp

from . import termination_fns


def cartpole(act: jnp.ndarray, next_obs: jnp.ndarray) -> jnp.ndarray:
  assert len(next_obs.shape) == len(act.shape) == 2

  return (~termination_fns.cartpole(act, next_obs)).astype(jnp.float32).reshape(-1)
