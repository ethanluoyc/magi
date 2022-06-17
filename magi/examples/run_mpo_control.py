"""Run MPO on dm_control (state observation)."""
import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from dm_control import suite  # pytype: disable=import-error
import jax
import numpy as np
import tensorflow as tf

from magi import wrappers as magi_wrappers
from magi.agents import mpo
from magi.utils import loggers

FLAGS = flags.FLAGS
flags.DEFINE_string('domain_name', 'cartpole', 'dm_control domain')
flags.DEFINE_string('task_name', 'swingup', 'dm_control task')
flags.DEFINE_bool('wandb', False, 'whether to log result to wandb')
flags.DEFINE_string('wandb_project', 'magi', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'ethanluoyc', 'wandb project entity')
flags.DEFINE_integer('num_steps', int(1e6), 'Random seed.')
flags.DEFINE_integer('eval_every', int(5000), 'Random seed.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def load_env(domain_name, task_name, seed):
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
  env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed)
  eval_env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed + 1)
  spec = specs.make_environment_spec(env)
  exp_name = (
      f'mpo-{FLAGS.domain_name}_{FLAGS.task_name}_{FLAGS.seed}_{int(time.time())}'
  )
  agent_networks = mpo.make_networks(spec)
  agent = mpo.MPO(
      environment_spec=spec,
      networks=agent_networks,
      seed=FLAGS.seed,
      config=mpo.MPOConfig(),
      logger=loggers.make_logger(
          'agent',
          use_wandb=FLAGS.wandb,
          log_frequency=1000,
          wandb_kwargs={
              'project': FLAGS.wandb_project,
              'entity': FLAGS.wandb_entity,
              'name': exp_name,
              'config': FLAGS,
          },
      ),
  )
  eval_actor = agent.builder.make_actor(
      random_key=jax.random.PRNGKey(FLAGS.seed + 1),
      policy_network=mpo.apply_policy_and_sample(
          agent_networks, spec.actions, eval_mode=True),
      variable_source=agent,
  )
  eval_loop = acme.EnvironmentLoop(
      eval_env,
      eval_actor,
      logger=loggers.make_logger(
          'eval_loop', use_wandb=FLAGS.wandb, log_frequency=1),
  )

  train_loop = acme.EnvironmentLoop(
      env,
      agent,
      logger=loggers.make_logger(
          label='environment_loop', use_wandb=FLAGS.wandb),
  )
  assert FLAGS.num_steps % FLAGS.eval_every == 0
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    eval_loop.run(num_episodes=5)
    train_loop.run(num_steps=FLAGS.eval_every)
  eval_loop.run(num_episodes=5)


if __name__ == '__main__':
  tf.config.set_visible_devices([], 'GPU')
  app.run(main)
