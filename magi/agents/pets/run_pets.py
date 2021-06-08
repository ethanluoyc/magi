#!/usr/bin/env python
# coding: utf-8

from absl import app
from acme import environment_loop
from acme import specs
from acme.utils import loggers
from acme import wrappers
from gym.wrappers.time_limit import TimeLimit

from magi.agents.pets import builder
from magi.agents.pets.cartpole import CartpoleEnv
from magi.agents.pets import cartpole_config


def make_environment():
  """Creates an OpenAI Gym environment."""
  # Load the gym environment.
  environment = CartpoleEnv()
  environment.seed(0)
  environment = TimeLimit(environment, cartpole_config.TIME_LIMIT)
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def main(unused_argv):
  del unused_argv
  environment = make_environment()
  environment_spec = specs.make_environment_spec(environment)
  config = cartpole_config
  agent = builder.make_agent(environment_spec,
                             config.cost_fn,
                             config.obs_preproc,
                             config.obs_postproc,
                             config.targ_proc,
                             "cem",
                             hidden_sizes=(128, 128, 128))

  env_loop_logger = loggers.TerminalLogger(label="environment_loop")
  env_loop = environment_loop.EnvironmentLoop(environment,
                                              agent,
                                              logger=env_loop_logger)
  env_loop.run(num_episodes=500)


if __name__ == "__main__":
  app.run(main)
