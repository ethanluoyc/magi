import math
import jax.numpy as jnp


def cartpole(act: jnp.ndarray, next_obs: jnp.ndarray) -> jnp.ndarray:
  del act
  assert len(next_obs.shape) == 2

  x, theta = next_obs[:, 0], next_obs[:, 2]

  x_threshold = 2.4
  theta_threshold_radians = 12 * 2 * math.pi / 360
  not_done = ((x > -x_threshold) * (x < x_threshold) *
              (theta > -theta_threshold_radians) * (theta < theta_threshold_radians))
  done = ~not_done
  return done
