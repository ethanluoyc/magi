import dataclasses
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from magi.environments import termination_fns


@dataclasses.dataclass
class Config:
    task_horizon: int = 1000
    hidden_sizes: Sequence[int] = (200, 200, 200)
    population_size: int = 500
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
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
        """Get goal for the task."""
        del env
        return None

    @staticmethod
    def obs_preproc(obs):
        """Preprocess the observation."""
        raise NotImplementedError()

    @staticmethod
    def obs_postproc(obs, pred):
        """Postprocess the predicted next observation."""
        raise NotImplementedError()

    @staticmethod
    def targ_proc(obs, next_obs):
        """Process the obseravtion for training ensembles."""
        raise NotImplementedError()

    @staticmethod
    def reward_fn(obs, acs, goal):
        """Reward function for environment."""
        raise NotImplementedError()

    @staticmethod
    def termination_fn(obs, act, goal):
        """Termination function for environment.
        Default to no termination.
        """
        del goal
        return termination_fns.no_termination(act, obs)
