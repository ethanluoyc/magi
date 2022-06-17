"""Run CRR on D4RL."""
from absl import app
from absl import flags
from absl import logging
from acme import specs
from acme import wrappers
import d4rl  # type: ignore
import gym
import jax
import numpy as np
import tensorflow as tf
import wandb

from magi.agents import crr
from magi.examples.offline import d4rl_dataset
from magi.utils import loggers

FLAGS = flags.FLAGS
flags.DEFINE_string('policy', 'CRR (JAX)', 'Policy name')
flags.DEFINE_string('env', 'hopper-medium-v0', 'OpenAI gym environment name')
flags.DEFINE_integer('seed', 0, 'seed')
flags.DEFINE_integer('log_freq', 500, 'log frequency')
flags.DEFINE_integer('eval_freq', int(5e3), 'evaluation frequency')
flags.DEFINE_integer('max_timesteps', int(1e6), 'maximum number of steps')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('eval_episodes', 10, 'number of evaluation episodes')
flags.DEFINE_float('discount', 0.99, 'discount')
flags.DEFINE_bool('normalize', False, 'normalize data')
flags.DEFINE_bool('wandb', False, 'whether to use W&B')


def evaluate(actor,
             env_name,
             seed,
             mean,
             std,
             seed_offset=100,
             eval_episodes=10):
  """Evaluate the policy
    Runs policy for X episodes and returns average reward.
    A fixed seed is used for the eval environment
    """
  eval_env = make_environment(env_name)
  eval_env.seed(seed + seed_offset)
  actor.update(wait=True)
  avg_reward = 0.0
  for _ in range(eval_episodes):
    timestep = eval_env.reset()
    actor.observe_first(timestep)
    while not timestep.last():
      obs = (np.array(timestep.observation) - mean) / std
      action = actor.select_action(obs)
      timestep = eval_env.step(action)
      actor.observe(action, timestep)
      avg_reward += timestep.reward

  avg_reward /= eval_episodes
  d4rl_score = eval_env.get_normalized_score(avg_reward)

  logging.info('---------------------------------------')
  logging.info('Evaluation over %d episodes: %.3f', eval_episodes, d4rl_score)
  logging.info('---------------------------------------')
  return d4rl_score


def make_environment(name):
  environment = gym.make(name)
  environment = wrappers.GymWrapper(environment)
  return wrappers.SinglePrecisionWrapper(environment)


def main(_):
  # Disable TF GPU

  tf.config.set_visible_devices([], 'GPU')
  if FLAGS.wandb:
    wandb.init(project='magi', entity='ethanluoyc', name='CRR (JAX)')
  logging.info('---------------------------------------')
  logging.info('Policy: %s, Env: %s, Seed: %s', FLAGS.policy, FLAGS.env,
               FLAGS.seed)
  logging.info('---------------------------------------')

  np.random.seed(FLAGS.seed)
  env = make_environment(FLAGS.env)
  environment_spec = specs.make_environment_spec(env)
  env.seed(FLAGS.seed)
  config = crr.CRRConfig(discount=FLAGS.discount, batch_size=FLAGS.batch_size)
  agent_networks = crr.make_networks(environment_spec)
  data = d4rl.qlearning_dataset(env)

  def learner_logger_fn():
    return loggers.make_logger(
        'learner',
        log_frequency=FLAGS.log_freq,
        use_wandb=FLAGS.wandb,
        wandb_kwargs={'config': FLAGS},
    )

  builder = crr.CRRBuilder(config, logger_fn=learner_logger_fn)
  if FLAGS.normalize:
    data, mean, std = d4rl_dataset.normalize_obs(data)
  else:
    mean, std = 0, 1
  data_iterator = d4rl_dataset.make_tf_data_iterator(
      data, batch_size=FLAGS.batch_size).as_numpy_iterator()
  learner_key, actor_key = jax.random.split(jax.random.PRNGKey(FLAGS.seed))

  learner = builder.make_learner(
      networks=agent_networks, random_key=learner_key, dataset=data_iterator)
  evaluator_network = crr.apply_policy_and_sample(
      agent_networks, eval_mode=True)
  evaluator = builder.make_actor(
      evaluator_network, actor_key, variable_source=learner)
  evaluations = []
  for t in range(int(FLAGS.max_timesteps)):
    learner.step()
    # Evaluate episode
    if (t + 1) % FLAGS.eval_freq == 0:
      logging.info('Time steps: %d', t + 1)
      evaluations.append(
          evaluate(
              evaluator,
              FLAGS.env,
              FLAGS.seed,
              mean,
              std,
              eval_episodes=FLAGS.eval_episodes,
          ))
      if FLAGS.wandb:
        wandb.log({'step': t, 'eval_returns': evaluations[-1]})


if __name__ == '__main__':
  FLAGS.logtostderr = True
  app.run(main)
