#!/usr/bin/env python
# coding: utf-8

from absl import app
from absl import flags
from acme import specs
from acme import wrappers
from acme.utils import loggers
from gym import wrappers as gym_wrappers
import jax
import numpy as np

from magi.agents.pets import builder
from magi.agents.pets import configs
from magi.environments.cartpole import CartpoleEnv
from magi.environments.pets_cheetah import HalfCheetahEnv
from magi.environments.pusher import PusherEnv
from magi.environments.reacher import Reacher3DEnv

FLAGS = flags.FLAGS
flags.DEFINE_bool("wandb", False, "whether to log result to wandb")
flags.DEFINE_string("wandb_project", "magi", "wandb project name")
flags.DEFINE_string("wandb_entity", "ethanluoyc", "wandb project entity")
flags.DEFINE_string("env", "halfcheetah", "environment")
flags.DEFINE_integer("num_episodes", int(100), "Number of episodes.")
flags.DEFINE_integer("seed", 0, "Random seed.")


def make_environment(name, task_horizon, seed):
    """Creates an OpenAI Gym environment."""
    # Load the gym environment.
    if name == "reacher":
        environment = Reacher3DEnv()
    elif name == "pusher":
        environment = PusherEnv()
    elif name == "halfcheetah":
        environment = HalfCheetahEnv()
    elif name == "cartpole":
        environment = CartpoleEnv()
    else:
        raise ValueError("Unknown environment")
    environment = gym_wrappers.TimeLimit(environment, task_horizon)
    environment.seed(seed)
    environment = wrappers.GymWrapper(environment)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment


def get_config(env_name):
    if env_name == "reacher":
        return configs.ReacherConfig()
    elif env_name == "pusher":
        return configs.PusherConfig()
    elif env_name == "halfcheetah":
        return configs.HalfCheetahConfig()
    elif env_name == "cartpole":
        return configs.CartPoleConfig()
    else:
        raise ValueError("Unknown environment")


def main(unused_argv):
    del unused_argv
    np.random.seed(FLAGS.seed)
    rng = np.random.default_rng(FLAGS.seed + 1)
    config = get_config(FLAGS.env)
    environment = make_environment(
        FLAGS.env, config.task_horizon, int(rng.integers(0, 2 ** 32))
    )
    environment_spec = specs.make_environment_spec(environment)
    print("observation spec", environment_spec.observations.shape)
    print("action_spec", environment_spec.actions.shape)
    agent = builder.make_agent(
        environment_spec,
        config.cost_fn,
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

    logger = loggers.TerminalLogger(label="environment_loop")
    for episode in range(FLAGS.num_episodes):
        timestep = environment.reset()
        goal = config.get_goal(environment)
        agent.update_goal(goal)
        agent.observe_first(timestep)
        episode_return = 0.0
        num_steps = 0
        while not timestep.last():
            action = agent.select_action(observation=timestep.observation)
            timestep = environment.step(action)
            num_steps += 1
            agent.observe(action, next_timestep=timestep)
            agent.update()
            episode_return += timestep.reward
        logger.write({"episode": episode, "episode_return": episode_return})


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
