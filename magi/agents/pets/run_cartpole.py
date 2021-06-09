#!/usr/bin/env python
# coding: utf-8

from absl import app
from acme import environment_loop
from acme import specs
from acme.utils import loggers
from acme import wrappers
import jax

from magi.agents.pets import builder
from gym.wrappers import TimeLimit
from magi.environments.cartpole_continuous import CartPoleEnv
from magi.environments import reward_fns
from magi.environments import termination_fns
# import jax.numpy as jnp

SEED = 2

def obs_preproc(obs):
  return obs
  # return jnp.concatenate(
  #     [jnp.sin(obs[:, 1:2]),
  #      jnp.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)


def obs_postproc(obs, pred):
  return obs + pred


def targ_proc(obs, next_obs):
  return next_obs - obs


def make_environment():
  """Creates an OpenAI Gym environment."""
  # Load the gym environment.
  environment = CartPoleEnv()
  environment = TimeLimit(environment, 200)
  environment.seed(SEED)
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def main(unused_argv):
  del unused_argv
  environment = make_environment()
  environment_spec = specs.make_environment_spec(environment)
  agent = builder.make_agent(environment_spec, lambda x, a, goal: -reward_fns.cartpole(a, x),
                             lambda x, a, goal: termination_fns.cartpole(a, x), obs_preproc,
                             obs_postproc, targ_proc,
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
