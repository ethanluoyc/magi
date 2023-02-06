"""Base configuration for running PETS experiments."""
import jax
import ml_collections
from environments import termination_fns


def get_base_config():
    config = ml_collections.ConfigDict()
    config.task_horizon = 1000
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

    config.get_goal = lambda env: None
    config.obs_preproc = lambda obs: obs
    config.obs_postproc = None
    config.targ_proc = None
    config.reward_fn = None
    config.termination_fn = lambda obs, act, goal: termination_fns.no_termination(
        act, obs
    )

    return config
