import math

import jax.numpy as jnp


def cartpole(act: jnp.ndarray, next_obs: jnp.ndarray) -> jnp.ndarray:
    del act
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    return done


def no_termination(act: jnp.ndarray, next_obs: jnp.ndarray) -> jnp.ndarray:
    del act
    assert len(next_obs.shape) == 2

    done = jnp.zeros((len(next_obs),), dtype=jnp.bool_)
    return done


def inverted_pendulum(act: jnp.ndarray, next_obs: jnp.ndarray) -> jnp.ndarray:
    assert len(next_obs.shape) == 2

    not_done = jnp.isfinite(next_obs).all(-1) * (next_obs[:, 1].abs() <= 0.2)
    done = ~not_done

    return done
