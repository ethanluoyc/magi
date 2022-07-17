"""Run TD3 on Gym Mujoco environments."""
import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
import gym
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from magi.agents import td3
from magi.utils import loggers

tfd = tfp.distributions

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'HalfCheetah-v2', 'Gym environment name')
flags.DEFINE_bool('wandb', False, 'whether to log result to wandb')
flags.DEFINE_string('wandb_project', 'magi', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'ethanluoyc', 'wandb project entity')
flags.DEFINE_integer('num_steps', int(1e6), 'Random seed.')
flags.DEFINE_integer('num_initial_steps', int(25e3), 'Random seed.')
flags.DEFINE_integer('eval_every', 10000, 'Evaluate every n steps')
flags.DEFINE_integer('eval_episodes', 10, 'Evaluate every n steps')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def load_env(env_name: str, seed: int):
  env = gym.make(env_name)
  env = wrappers.wrap_all(
      env,
      [
          wrappers.GymWrapper,
          wrappers.CanonicalSpecWrapper,
          wrappers.SinglePrecisionWrapper,
      ],
  )
  env.seed(seed)
  return env


def main(_):
  np.random.seed(FLAGS.seed)
  env = load_env(FLAGS.env, FLAGS.seed)
  environment_spec = specs.make_environment_spec(env)
  exp_name = f'td3_gym_{FLAGS.env}_{FLAGS.seed}_{int(time.time())}'

  td3_config = td3.TD3Config(min_replay_size=FLAGS.num_initial_steps)
  agent_networks = td3.make_networks(environment_spec)
  agent = td3.TD3Agent(
      environment_spec=environment_spec,
      networks=agent_networks,
      config=td3_config,
      random_key=jax.random.PRNGKey(FLAGS.seed),
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

  train_loop = acme.EnvironmentLoop(
      env,
      agent,
      logger=loggers.make_logger(label='train_loop', use_wandb=FLAGS.wandb),
  )
  evaluator = agent.builder.make_actor(
      jax.random.PRNGKey(FLAGS.seed + 10),
      td3.apply_policy_sample(agent_networks, eval_mode=True),
      environment_spec,
      variable_source=agent,
  )
  eval_env = load_env(FLAGS.env, FLAGS.seed + 1000)
  eval_loop = acme.EnvironmentLoop(
      eval_env,
      evaluator,
      label='eval_loop',
      logger=loggers.make_logger(label='eval_loop', use_wandb=FLAGS.wandb),
  )
  assert FLAGS.num_steps % FLAGS.eval_every == 0
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    eval_loop.run(num_episodes=FLAGS.eval_episodes)
    train_loop.run(num_steps=FLAGS.eval_every)
  eval_loop.run(num_episodes=FLAGS.eval_episodes)


if __name__ == '__main__':
  app.run(main)
