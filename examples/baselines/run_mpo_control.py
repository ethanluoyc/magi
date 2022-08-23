"""Run MPO on dm_control (state observation)."""
import time

from absl import app
from absl import flags
from acme import wrappers
from dm_control import suite  # pytype: disable=import-error
import numpy as np
import tensorflow as tf

from magi import experiments
from magi import wrappers as magi_wrappers
from magi.agents import mpo

FLAGS = flags.FLAGS
flags.DEFINE_string('domain_name', 'cartpole', 'dm_control domain')
flags.DEFINE_string('task_name', 'swingup', 'dm_control task')
flags.DEFINE_bool('wandb', False, 'whether to log result to wandb')
flags.DEFINE_string('wandb_project', 'magi', 'wandb project name')
flags.DEFINE_string('wandb_entity', None, 'wandb project entity')
flags.DEFINE_integer('num_steps', int(1e6), 'Random seed.')
flags.DEFINE_integer('eval_every', int(5000), 'Random seed.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def make_environment(domain_name: str, task_name: str, seed: int):
  env = suite.load(
      domain_name=domain_name,
      task_name=task_name,
      environment_kwargs={'flat_observation': True},
      task_kwargs={'random': seed},
  )
  env = wrappers.CanonicalSpecWrapper(env)
  env = magi_wrappers.TakeKeyWrapper(env, 'observations')
  env = wrappers.SinglePrecisionWrapper(env)
  return env


def main(_):
  np.random.seed(FLAGS.seed)
  exp_name = (
      f'mpo-{FLAGS.domain_name}_{FLAGS.task_name}_{FLAGS.seed}_{int(time.time())}'
  )
  logger_factory = experiments.LoggerFactory(
      log_to_wandb=FLAGS.wandb,
      wandb_kwargs={
          'project': FLAGS.wandb_project,
          'entity': FLAGS.wandb_entity,
          'name': exp_name,
          'config': FLAGS,
      },
      learner_time_delta=5.0,
      evaluator_time_delta=0.0)

  builder = mpo.MPOBuilder(config=mpo.MPOConfig())
  experiment = experiments.ExperimentConfig(
      builder,
      mpo.make_networks,
      lambda seed: make_environment(FLAGS.domain_name, FLAGS.task_name, seed),
      max_num_actor_steps=FLAGS.num_steps,
      seed=FLAGS.seed,
      logger_factory=logger_factory)
  experiments.run_experiment(
      experiment, eval_every=FLAGS.eval_every, num_eval_episodes=5)


if __name__ == '__main__':
  tf.config.set_visible_devices([], 'GPU')
  app.run(main)
