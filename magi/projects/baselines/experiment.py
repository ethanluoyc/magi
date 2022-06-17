"""Run offline learning experiments."""

import functools
import os
from typing import Iterator, Tuple

from absl import app
from absl import flags
from absl import logging
from acme import types
from acme import wrappers
import gym
import jax
from ml_collections import config_flags
import numpy as np
import wandb
import yaml

from magi.projects.baselines import dataset_utils

config_flags.DEFINE_config_file('config')
flags.DEFINE_string('workdir', None, 'Where to save log results')
flags.mark_flags_as_required(['config', 'workdir'])

FLAGS = flags.FLAGS


def evaluate(actor, environment, eval_episodes=10):
  actor.update(wait=True)
  avg_reward = 0.0
  for _ in range(eval_episodes):
    timestep = environment.reset()
    actor.observe_first(timestep)
    while not timestep.last():
      action = actor.select_action(timestep.observation)
      timestep = environment.step(action)
      actor.observe(action, timestep)
      avg_reward += timestep.reward

  avg_reward /= eval_episodes
  d4rl_score = environment.get_normalized_score(avg_reward)

  logging.info('---------------------------------------')
  logging.info('Evaluation over %d episodes: %.3f', eval_episodes, d4rl_score)
  logging.info('---------------------------------------')
  return d4rl_score


def normalize(dataset):

  trajs = dataset_utils.split_into_trajectories(
      dataset.observations,
      dataset.actions,
      dataset.rewards,
      dataset.masks,
      dataset.dones_float,
      dataset.next_observations,
  )

  def compute_returns(traj):
    episode_return = 0
    for _, _, rew, _, _, _ in traj:
      episode_return += rew

    return episode_return

  trajs.sort(key=compute_returns)

  dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
  dataset.rewards *= 1000.0


def _make_dataset_iterator(dataset, batch_size: int):
  while True:
    batch = dataset.sample(batch_size)
    yield batch


def make_env_and_dataset(
    env_name: str, seed: int,
    batch_size: int) -> Tuple[gym.Env, Iterator[types.Transition]]:
  env = gym.make(env_name)
  env.seed(seed)
  env = wrappers.wrap_all(
      env,
      [
          wrappers.GymWrapper,
          wrappers.SinglePrecisionWrapper,
      ],
  )

  dataset = dataset_utils.D4RLDataset(env)

  if 'antmaze' in env_name:
    dataset.rewards = (dataset.rewards - 0.5) * 4.0
  elif 'halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name:
    normalize(dataset)

  return env, _make_dataset_iterator(dataset, batch_size)


def run(main):
  jax.config.config_with_absl()
  app.run(functools.partial(_run_main, main=main))


def _run_main(argv, *, main):
  del argv

  # Create working directory
  os.makedirs(FLAGS.workdir, exist_ok=True)

  # Fix global numpy random seed
  np.random.seed(FLAGS.config.seed)

  # Initialilize wandb if needed
  if FLAGS.config.log_to_wandb:
    wandb.init(config=FLAGS.config.to_dict(), dir=FLAGS.workdir)

  # Save configuration
  with open(os.path.join(FLAGS.workdir, 'config.yaml'), 'wt') as f:
    yaml.dump(FLAGS.config.to_dict(), f)

  main(config=FLAGS.config, workdir=FLAGS.workdir)
