"""Configuration for running PETS cartpole experiments."""
import base_config
import jax
import jax.numpy as jnp


def get_config():
    config = base_config.get_base_config()
    config.task_horizon = 200
    config.hidden_sizes = (200, 200, 200)
    config.population_size = 500
    config.activation = jax.nn.silu
    config.planning_horizon = 15
    config.cem_alpha = 0.1
    config.cem_elite_frac = 0.1
    config.cem_return_mean_elites = True
    config.weight_decay = 5e-5
    config.lr = 1e-3
    config.min_delta = 0.01
    config.num_ensembles = 5
    config.num_particles = 20
    config.num_epochs = 50
    config.patience = 50

    def obs_preproc(obs):
        return jnp.concatenate(
            [jnp.sin(obs[:, 1:2]), jnp.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1
        )

    def obs_postproc(obs, pred):
        return obs + pred

    def targ_proc(obs, next_obs):
        return next_obs - obs

    def obs_cost_fn(obs):
        return -jnp.exp(
            -jnp.sum(
                jnp.square(_get_ee_pos(obs) - jnp.array([0.0, 0.6])),
                axis=1,
            )
            / 0.6**2
        )

    def ac_cost_fn(acs):
        return 0.01 * jnp.sum(jnp.square(acs), axis=1)

    def reward_fn(obs, acs, goal):
        del goal
        return -(obs_cost_fn(obs) + ac_cost_fn(acs))

    def _get_ee_pos(obs):
        x0, theta = obs[:, :1], obs[:, 1:2]
        return jnp.concatenate(
            [x0 - 0.6 * jnp.sin(theta), -0.6 * jnp.cos(theta)], axis=1
        )

    config.env_name = "cartpole"
    config.obs_preproc = obs_preproc
    config.obs_postproc = obs_postproc
    config.targ_proc = targ_proc
    config.reward_fn = reward_fn
    return config
