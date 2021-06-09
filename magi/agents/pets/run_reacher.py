#!/usr/bin/env python
# coding: utf-8

import sys

from absl import app
from absl import flags
from acme import specs
from acme import wrappers
from gym.wrappers import TimeLimit
import jax
import numpy as np

from magi.agents.pets import builder
from magi.agents.pets.configs import reacher as config
from magi.environments import termination_fns
from magi.environments.reacher import Reacher3DEnv

FLAGS = flags.FLAGS
flags.DEFINE_bool('wandb', False, 'whether to log result to wandb')
flags.DEFINE_string('wandb_project', 'magi', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'ethanluoyc', 'wandb project entity')
flags.DEFINE_integer('num_episodes', int(100), 'Number of episodes.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def make_environment(seed):
  """Creates an OpenAI Gym environment."""
  # Load the gym environment.
  environment = Reacher3DEnv()
  environment = TimeLimit(environment, config.TASK_HORIZON)
  environment.seed(seed)
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def main(unused_argv):
  np.random.seed(FLAGS.seed)
  del unused_argv
  # if FLAGS.wandb:
  #   import wandb  # pylint: disable=import-outside-toplevel
  #   wandb.init(project=FLAGS.wandb_project,
  #              entity=FLAGS.wandb_entity,
  #              name=f'pets_halfcheetah_{FLAGS.seed}',
  #              config=FLAGS)
  rng = np.random.default_rng(FLAGS.seed)
  environment = make_environment(int(rng.integers(0, sys.maxsize + 1, dtype=np.int64)))
  environment_spec = specs.make_environment_spec(environment)
  agent = builder.make_agent(environment_spec,
                             lambda x, a, goal: config.cost_fn(x, a, goal),
                             lambda x, a, goal: termination_fns.no_termination(a, x),
                             config.obs_preproc,
                             config.obs_postproc,
                             config.targ_proc,
                             hidden_sizes=(200, 200, 200),
                             population_size=500,
                             activation=jax.nn.swish,
                             time_horizon=25,
                             cem_alpha=0.1,
                             cem_elite_frac=0.1,
                             cem_return_mean_elites=True,
                             weight_decay=1e-4,
                             lr=1e-3,
                             min_delta=0.01,
                             num_ensembles=5,
                             num_particles=20,
                             num_epochs=5,
                             seed=rng.integers(-sys.maxsize - 1,
                                               sys.maxsize + 1,
                                               dtype=np.int64),
                             patience=5)

  for episode in range(FLAGS.num_episodes):
    timestep = environment.reset()
    agent._actor.update_goal(config.get_goal(environment))
    agent.observe_first(timestep)
    episode_return = 0.
    while not timestep.last():
      action = agent.select_action(observation=timestep.observation)
      timestep = environment.step(action)
      agent.observe(action, timestep)
      agent.update()
      episode_return += timestep.reward
    print(episode, episode_return)

  # if FLAGS.wandb:
  #   wandb.finish()


if __name__ == '__main__':
  app.run(main)
