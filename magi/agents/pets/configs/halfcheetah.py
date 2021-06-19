import dataclasses
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from magi.agents.pets.configs.default import Config


@dataclasses.dataclass
class HalfCheetahConfig(Config):
    task_horizon: int = 1000
    hidden_sizes: Sequence[int] = (200, 200, 200, 200)
    population_size: int = 350
    activation: Callable = jax.nn.silu
    planning_horizon: int = 30
    cem_alpha: float = 0.1
    cem_elite_frac: float = 0.1
    cem_return_mean_elites: bool = True
    weight_decay: float = 3e-5
    lr: float = 2e-4
    min_delta: float = 0.01
    num_ensembles: int = 5
    num_particles: int = 20
    num_epochs: int = 25
    patience: int = 25

    def get_goal(self, env):
        # Cheetah does not have a goal
        del env

    @staticmethod
    def obs_preproc(state):
        assert isinstance(state, jnp.ndarray)
        assert state.ndim in (1, 2, 3)
        d1 = state.ndim == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = jnp.expand_dims(state, 0)
        ret = jnp.concatenate(
            [
                state[..., 1:2],
                jnp.sin(state[..., 2:3]),
                jnp.cos(state[..., 2:3]),
                state[..., 3:],
            ],
            axis=state.ndim - 1,
        )
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    @staticmethod
    def obs_postproc(obs, pred):
        return jnp.concatenate([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)

    @staticmethod
    def targ_proc(obs, next_obs):
        assert len(obs.shape) == 2
        assert len(next_obs.shape) == 2
        return jnp.concatenate([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], axis=1)

    @staticmethod
    def obs_cost_fn(obs):
        return -obs[:, 0]

    @staticmethod
    def ac_cost_fn(acs):
        return 0.1 * (acs ** 2).sum(axis=1)

    @staticmethod
    def cost_fn(obs, acs, goal):
        del goal
        return HalfCheetahConfig.obs_cost_fn(obs) + HalfCheetahConfig.ac_cost_fn(acs)
