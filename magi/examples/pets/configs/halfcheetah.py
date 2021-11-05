"""Configuration for running PETS halfcheetah experiments."""
import jax
import jax.numpy as jnp

from magi.examples.pets.configs import base


def get_config():
    config = base.get_base_config()
    config.task_horizon = 1000
    config.hidden_sizes = (200, 200, 200, 200)
    config.population_size = 350
    config.activation = jax.nn.silu
    config.planning_horizon = 30
    config.cem_alpha = 0.1
    config.cem_elite_frac = 0.1
    config.cem_return_mean_elites = True
    config.weight_decay = 3e-5
    config.lr = 2e-4
    config.min_delta = 0.01
    config.num_ensembles = 5
    config.num_particles = 20
    config.num_epochs = 25
    config.patience = 25

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

    def obs_postproc(obs, pred):
        return jnp.concatenate([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)

    def targ_proc(obs, next_obs):
        assert len(obs.shape) == 2
        assert len(next_obs.shape) == 2
        return jnp.concatenate([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], axis=1)

    def obs_cost_fn(obs):
        return -obs[:, 0]

    def ac_cost_fn(acs):
        return 0.1 * (acs ** 2).sum(axis=1)

    def reward_fn(obs, acs, goal):
        del goal
        return -(obs_cost_fn(obs) + ac_cost_fn(acs))

    config.env_name = "halfcheetah"
    config.obs_preproc = obs_preproc
    config.obs_postproc = obs_postproc
    config.targ_proc = targ_proc
    config.reward_fn = reward_fn
    return config
