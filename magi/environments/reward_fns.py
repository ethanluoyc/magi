import jax.numpy as jnp

from . import termination_fns


def cartpole(act: jnp.ndarray, next_obs: jnp.ndarray) -> jnp.ndarray:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).astype(jnp.float32).reshape(-1)


def halfcheetah(act: jnp.ndarray, next_obs: jnp.ndarray) -> jnp.ndarray:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * jnp.square(act).sum(axis=1)
    reward_run = next_obs[:, 0] - 0.0 * jnp.square(next_obs[:, 2])
    return (reward_run + reward_ctrl).reshape(-1)


def inverted_pendulum(act: jnp.ndarray, next_obs: jnp.ndarray) -> jnp.ndarray:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).astype(jnp.float32)
