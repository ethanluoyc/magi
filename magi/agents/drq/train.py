import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from acme.wrappers import gym_wrapper
import numpy as np

from magi.agents import drq
from magi.agents.drq import networks
from magi.agents.drq.agent import DrQConfig
from magi.agents.sac_ae.environment import make_dmc_env
from magi.utils import loggers

FLAGS = flags.FLAGS
flags.DEFINE_string('domain_name', 'cheetah', 'dm_control domain')
flags.DEFINE_string('task_name', 'run', 'dm_control task')
flags.DEFINE_bool('wandb', False, 'whether to log result to wandb')
flags.DEFINE_string('wandb_project', 'magi', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'ethanluoyc', 'wandb project entity')
flags.DEFINE_string('logdir', './logs', '')
flags.DEFINE_integer('num_steps', int(1e6), '')
flags.DEFINE_integer('eval_freq', 5000, '')
flags.DEFINE_integer('eval_episodes', 10, '')
flags.DEFINE_integer('frame_stack', 3, '')
flags.DEFINE_integer('action_repeat', 4, '')
flags.DEFINE_integer('max_replay_size', 100_000, 'Minimum replay size')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('seed', 42, 'Random seed.')


def load_env(domain_name, task_name, seed, frame_stack, action_repeat):
  # TODO(yl): Remove make_dmc_env and construct environment from dm_control directly.
  env = make_dmc_env(domain_name,
                     task_name,
                     action_repeat,
                     n_frames=frame_stack,
                     image_size=84)
  env.seed(seed)
  return wrappers.SinglePrecisionWrapper(gym_wrapper.GymWrapper(env))


def main(_):
  np.random.seed(FLAGS.seed)
  if FLAGS.wandb:
    import wandb  # pylint: disable=import-outside-toplevel
    experiment_name = (f'drq-{FLAGS.domain_name}-{FLAGS.task_name}_'
                       f'{FLAGS.seed}_{int(time.time())}')
    wandb.init(project=FLAGS.wandb_project,
               entity=FLAGS.wandb_entity,
               name=experiment_name,
               config=FLAGS,
               dir=FLAGS.logdir)
  env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed, FLAGS.frame_stack,
                 FLAGS.action_repeat)
  test_env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed + 1000,
                      FLAGS.frame_stack, FLAGS.action_repeat)
  spec = specs.make_environment_spec(env)
  network_spec = networks.make_default_networks(spec)

  agent = drq.DrQAgent(environment_spec=spec,
                       networks=network_spec,
                       config=DrQConfig(max_replay_size=FLAGS.max_replay_size,
                                        batch_size=FLAGS.batch_size,
                                        temperature_adam_b1=0.9,
                                        ),
                       seed=FLAGS.seed,
                       logger=loggers.make_logger(label='learner',
                                                  time_delta=60,
                                                  use_wandb=FLAGS.wandb))
  eval_actor = agent.make_actor(is_eval=True)

  loop = acme.EnvironmentLoop(env,
                              agent,
                              logger=loggers.make_logger(label='environment_loop',
                                                         time_delta=5.,
                                                         use_wandb=FLAGS.wandb))
  eval_loop = acme.EnvironmentLoop(test_env,
                                   eval_actor,
                                   logger=loggers.make_logger(label='eval',
                                                              time_delta=0,
                                                              use_wandb=FLAGS.wandb))
  for _ in range(FLAGS.num_steps // FLAGS.eval_freq):
    loop.run(num_steps=FLAGS.eval_freq)
    eval_actor.update(wait=True)
    eval_loop.run(num_episodes=FLAGS.eval_episodes)

  if FLAGS.wandb:
    wandb.finish()


if __name__ == '__main__':
  app.run(main)
