#!/usr/bin/env python
# coding: utf-8

import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from gym import wrappers as gym_wrappers
import jax
from ml_collections import config_flags
import numpy as np

from magi.agents.pets import builder
from magi.examples.pets.environments.cartpole_continuous import CartPoleEnv
from magi.utils import loggers

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config")
flags.mark_flag_as_required("config")
flags.DEFINE_integer("num_episodes", int(100), "Number of episodes.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_bool("wandb", False, "whether to log result to wandb")
flags.DEFINE_string("wandb_project", "", "wandb project name")
flags.DEFINE_string("wandb_entity", "", "wandb project entity")


def make_environment(seed, task_horizon):
    """Creates an OpenAI Gym environment."""
    # Load the gym environment.
    environment = CartPoleEnv()
    environment = gym_wrappers.TimeLimit(environment, task_horizon)
    environment.seed(seed)
    environment = wrappers.GymWrapper(environment)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment


def main(unused_argv):
    del unused_argv
    config = FLAGS.config
    np.random.seed(FLAGS.seed)
    rng = np.random.default_rng(FLAGS.seed + 1)
    environment = make_environment(int(rng.integers(0, 2 ** 32)), config.task_horizon)
    environment_spec = specs.make_environment_spec(environment)
    agent = builder.make_agent(
        environment_spec,
        config.reward_fn,
        config.termination_fn,
        config.obs_preproc,
        config.obs_postproc,
        config.targ_proc,
        hidden_sizes=config.hidden_sizes,
        population_size=config.population_size,
        activation=jax.nn.silu,
        planning_horizon=config.planning_horizon,
        cem_alpha=config.cem_alpha,
        cem_elite_frac=config.cem_elite_frac,
        cem_return_mean_elites=config.cem_return_mean_elites,
        weight_decay=config.weight_decay,
        lr=config.lr,
        min_delta=config.min_delta,
        num_ensembles=config.num_ensembles,
        num_particles=config.num_particles,
        num_epochs=config.num_epochs,
        seed=rng.integers(0, 2 ** 32),
        patience=config.patience,
    )

    logger = loggers.make_logger(
        "environment_loop",
        use_wandb=FLAGS.wandb,
        wandb_kwargs={
            "project": FLAGS.wandb_project,
            "entity": FLAGS.wandb_entity,
            "name": f"pets_cartpole_{FLAGS.seed}_{int(time.time())}",
            "config": FLAGS,
        },
    )
    env_loop = acme.EnvironmentLoop(environment, agent, logger=logger)
    env_loop.run(num_episodes=FLAGS.num_episodes)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
