import dataclasses
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from magi.agents.pets.configs.default import Config


@dataclasses.dataclass
class CartPoleConfig(Config):
    task_horizon: int = 200
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
    def get_goal(env):
        del env
        return None

    @staticmethod
    def obs_preproc(obs):
        return jnp.concatenate(
            [jnp.sin(obs[:, 1:2]), jnp.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1
        )

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        return -jnp.exp(
            -jnp.sum(
                jnp.square(CartPoleConfig._get_ee_pos(obs) - jnp.array([0.0, 0.6])),
                axis=1,
            )
            / 0.6 ** 2
        )

    @staticmethod
    def ac_cost_fn(acs):
        return 0.01 * jnp.sum(jnp.square(acs), axis=1)

    @staticmethod
    def cost_fn(obs, acs, goal):
        del goal
        return CartPoleConfig.obs_cost_fn(obs) + CartPoleConfig.ac_cost_fn(acs)

    @staticmethod
    def _get_ee_pos(obs):
        x0, theta = obs[:, :1], obs[:, 1:2]
        return jnp.concatenate(
            [x0 - 0.6 * jnp.sin(theta), -0.6 * jnp.cos(theta)], axis=1
        )
