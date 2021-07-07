#!/usr/bin/env python
# coding: utf-8
"""Example running PETS on continuous control environments."""

import time
from typing import Tuple

from absl import app
from absl import flags
from acme import specs
from acme import wrappers
import dm_env
from gym import wrappers as gym_wrappers
import jax
import numpy as np

from magi.agents.pets import builder
from magi.environments.pets_cartpole import CartpoleEnv
from magi.environments.pets_halfcheetah import HalfCheetahEnv
from magi.environments.pets_pusher import PusherEnv
from magi.environments.pets_reacher import Reacher3DEnv
from magi.examples.pets import configs
from magi.utils import loggers

FLAGS = flags.FLAGS
flags.DEFINE_bool("wandb", False, "whether to log result to wandb")
flags.DEFINE_string("wandb_project", "magi", "wandb project name")
flags.DEFINE_string("wandb_entity", "ethanluoyc", "wandb project entity")
flags.DEFINE_string("env", "halfcheetah", "environment")
flags.DEFINE_integer("num_episodes", int(100), "Number of episodes.")
flags.DEFINE_integer("seed", 0, "Random seed.")

ENV_CONFIG_MAP = {
    "reacher": (Reacher3DEnv, configs.ReacherConfig),
    "pusher": (PusherEnv, configs.PusherConfig),
    "halfcheetah": (HalfCheetahEnv, configs.HalfCheetahConfig),
    "cartpole": (CartpoleEnv, configs.CartPoleConfig),
}


def make_environment(name, seed) -> Tuple[dm_env.Environment, configs.Config]:
    """Creates an OpenAI Gym environment."""
    # Load the gym environment.
    try:
        env_cls, cfg_cls = ENV_CONFIG_MAP[name]
        environment = env_cls()
        cfg = cfg_cls()
    except KeyError as e:
        raise ValueError(f"Unknown environment {name}") from e
    else:
        environment = gym_wrappers.TimeLimit(environment, cfg.task_horizon)
        environment.seed(seed)
        environment = wrappers.GymWrapper(environment)
        environment = wrappers.SinglePrecisionWrapper(environment)
        return environment, cfg


def main(unused_argv):
    del unused_argv
    np.random.seed(FLAGS.seed)
    rng = np.random.default_rng(FLAGS.seed + 1)
    environment, config = make_environment(FLAGS.env, int(rng.integers(0, 2 ** 32)))
    environment_spec = specs.make_environment_spec(environment)
    print("observation spec", environment_spec.observations.shape)
    print("action_spec", environment_spec.actions.shape)
    agent = builder.make_agent(
        environment_spec,
        config.reward_fn,
        config.termination_fn,
        config.obs_preproc,
        config.obs_postproc,
        config.targ_proc,
        hidden_sizes=config.hidden_sizes,
        population_size=config.population_size,
        activation=config.activation,
        planning_horizon=config.planning_horizon,
        cem_alpha=config.cem_alpha,
        cem_elite_frac=config.cem_elite_frac,
        cem_iterations=5,
        cem_return_mean_elites=config.cem_return_mean_elites,
        weight_decay=config.weight_decay,
        lr=config.lr,
        min_delta=config.min_delta,
        num_ensembles=config.num_ensembles,
        num_particles=config.num_particles,
        num_epochs=config.num_epochs,
        seed=FLAGS.seed + 1024,
        patience=config.patience,
    )

    logger = loggers.make_logger(
        "environment_loop",
        use_wandb=FLAGS.wandb,
        wandb_kwargs={
            "project": FLAGS.wandb_project,
            "entity": FLAGS.wandb_entity,
            "name": f"pets-{FLAGS.env}_{FLAGS.seed}_{int(time.time())}",
            "config": FLAGS,
        },
    )
    total_num_steps = 0
    for episode in range(FLAGS.num_episodes):
        timestep = environment.reset()
        goal = config.get_goal(environment)
        agent.update_goal(goal)
        agent.observe_first(timestep)
        episode_return = 0.0
        episode_steps = 0
        while not timestep.last():
            action = agent.select_action(observation=timestep.observation)
            timestep = environment.step(action)
            episode_steps += 1
            agent.observe(action, next_timestep=timestep)
            agent.update()
            episode_return += timestep.reward
        total_num_steps += episode_steps
        logger.write(
            {
                "episodes": episode,
                "episode_return": episode_return,
                "episode_length": episode_steps,
                "steps": total_num_steps,
            }
        )

    if FLAGS.wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
