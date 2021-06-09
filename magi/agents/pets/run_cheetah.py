#!/usr/bin/env python
# coding: utf-8

import sys
from absl import app
from absl import flags
from acme import environment_loop
from acme import specs
from acme import wrappers
from gym.wrappers import TimeLimit
import jax
import jax.numpy as jnp
import numpy as np

from magi.agents.pets import builder
from magi.environments import termination_fns
from magi.environments.pets_cheetah import HalfCheetahEnv
from magi.utils import loggers

FLAGS = flags.FLAGS
flags.DEFINE_bool('wandb', False, 'whether to log result to wandb')
flags.DEFINE_string('wandb_project', 'magi', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'ethanluoyc', 'wandb project entity')
flags.DEFINE_integer('num_episodes', int(100), 'Number of episodes.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def obs_preproc(obs):
  return HalfCheetahEnv._preprocess_state_jnp(obs)


def obs_postproc(obs, pred):
  return jnp.concatenate([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)


def targ_proc(obs, next_obs):
  assert len(obs.shape) == 2
  assert len(next_obs.shape) == 2
  return jnp.concatenate([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], axis=1)


def obs_cost_fn(obs):
  return -obs[:, 0]


def ac_cost_fn(acs):
  return 0.1 * (acs**2).sum(axis=1)


def cost_fn(obs, acs):
  return obs_cost_fn(obs) + ac_cost_fn(acs)


def make_environment(seed):
  """Creates an OpenAI Gym environment."""
  # Load the gym environment.
  environment = HalfCheetahEnv()
  environment = TimeLimit(environment, 1000)
  environment.seed(seed)
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def main(unused_argv):
  del unused_argv
  if FLAGS.wandb:
    import wandb  # pylint: disable=import-outside-toplevel
    wandb.init(project=FLAGS.wandb_project,
               entity=FLAGS.wandb_entity,
               name=f'pets_halfcheetah_{FLAGS.seed}',
               config=FLAGS)
  np.random.seed(FLAGS.seed)
  rng = np.random.default_rng(FLAGS.seed)
  environment = make_environment(int(rng.integers(0, sys.maxsize + 1, dtype=np.int64)))
  environment_spec = specs.make_environment_spec(environment)
  agent = builder.make_agent(environment_spec,
                             lambda x, a, goal: cost_fn(x, a),
                             lambda x, a, goal: termination_fns.no_termination(a, x),
                             obs_preproc,
                             obs_postproc,
                             targ_proc,
                             hidden_sizes=(200, 200, 200, 200),
                             population_size=500,
                             activation=jax.nn.silu,
                             time_horizon=30,
                             cem_alpha=0.1,
                             cem_elite_frac=0.1,
                             cem_return_mean_elites=True,
                             weight_decay=3e-5,
                             lr=2e-4,
                             min_delta=0.01,
                             num_ensembles=5,
                             num_particles=20,
                             num_epochs=25,
                             seed=rng.integers(-sys.maxsize - 1,
                                               sys.maxsize + 1,
                                               dtype=np.int64),
                             patience=25)
  logger = loggers.make_logger(label='environment_loop',
                               time_delta=0.0,
                               use_wandb=FLAGS.wandb)
  env_loop = environment_loop.EnvironmentLoop(environment, agent, logger=logger)
  env_loop.run(num_episodes=FLAGS.num_episodes)
  if FLAGS.wandb:
    wandb.finish()


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
