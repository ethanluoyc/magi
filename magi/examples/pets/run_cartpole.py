#!/usr/bin/env python
# coding: utf-8

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from acme.utils import loggers
from gym.wrappers import TimeLimit
import jax
import numpy as np

from magi.agents.pets import builder
from magi.environments.cartpole_continuous import CartPoleEnv
from magi.examples.pets import configs

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", int(100), "Number of episodes.")
flags.DEFINE_integer("seed", 0, "Random seed.")


def make_environment(seed, task_horizon):
    """Creates an OpenAI Gym environment."""
    # Load the gym environment.
    environment = CartPoleEnv()
    environment = TimeLimit(environment, task_horizon)
    environment.seed(seed)
    environment = wrappers.GymWrapper(environment)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment


def main(unused_argv):
    del unused_argv
    config = configs.CartPoleContinuousConfig()
    np.random.seed(FLAGS.seed)
    rng = np.random.default_rng(FLAGS.seed + 1)
    environment = make_environment(int(rng.integers(0, 2 ** 32)), config.task_horizon)
    environment_spec = specs.make_environment_spec(environment)
    agent = builder.make_agent(
        environment_spec,
        config.cost_fn,
        config.termination_fn,
        config.obs_preproc,
        config.obs_postproc,
        config.targ_proc,
        hidden_sizes=config.hidden_sizes,
        population_size=config.population_size,
        activation=jax.nn.silu,
        planning_horizon=config.task_horizon,
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

    env_loop_logger = loggers.TerminalLogger(label="environment_loop")
    env_loop = acme.EnvironmentLoop(environment, agent, logger=env_loop_logger)
    env_loop.run(num_episodes=FLAGS.num_episodes)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
