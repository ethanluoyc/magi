import jax

from magi.environments import reward_fns
from magi.environments import termination_fns
from magi.examples.pets.configs import base


def get_config():
    config = base.get_base_config()
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
        # Consistent with mbrl to not use any transformation on inputs
        # return jnp.concatenate(
        #     [jnp.sin(obs[:, 1:2]),
        #      jnp.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)
        return obs

    def obs_postproc(obs, pred):
        return obs + pred

    def targ_proc(obs, next_obs):
        return next_obs - obs

    def reward_fn(x, a, goal):
        del goal
        return reward_fns.cartpole(a, x)

    def termination_fn(x, a, goal):
        del goal
        return termination_fns.cartpole(a, x)

    config.obs_preproc = obs_preproc
    config.obs_postproc = obs_postproc
    config.targ_proc = targ_proc
    config.reward_fn = reward_fn
    config.termination_fn = termination_fn
    return config
