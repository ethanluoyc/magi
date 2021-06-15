#!/usr/bin/env python
# coding: utf-8

from absl import app
from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.utils import loggers
from gym.wrappers import TimeLimit
import jax

from magi.agents.pets import builder
from magi.agents.pets.configs import cartpole_continuous as config
from magi.environments.cartpole_continuous import CartPoleEnv

SEED = 2


def make_environment():
  """Creates an OpenAI Gym environment."""
  # Load the gym environment.
  environment = CartPoleEnv()
  environment = TimeLimit(environment, config.TASK_HORIZON)
  environment.seed(SEED)
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def main(unused_argv):
  del unused_argv
  environment = make_environment()
  environment_spec = specs.make_environment_spec(environment)
  agent = builder.make_agent(environment_spec,
                             config.cost_fn,
                             config.termination_fn,
                             config.obs_preproc,
                             config.obs_postproc,
                             config.targ_proc,
                             hidden_sizes=(200, 200, 200),
                             population_size=500,
                             activation=jax.nn.silu,
                             time_horizon=15,
                             cem_alpha=0.1,
                             cem_elite_frac=0.1,
                             cem_return_mean_elites=True,
                             weight_decay=5e-5,
                             lr=1e-3,
                             min_delta=0.01,
                             num_ensembles=5,
                             num_particles=20,
                             num_epochs=50,
                             seed=SEED,
                             patience=50)

  env_loop_logger = loggers.TerminalLogger(label="environment_loop")
  env_loop = environment_loop.EnvironmentLoop(environment,
                                              agent,
                                              logger=env_loop_logger)
  env_loop.run(num_episodes=500)


if __name__ == "__main__":
  app.run(main)
