from os import environ
import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from acme.utils import counting
from dm_control import suite  # pytype: disable=import-error
from dm_control.suite.wrappers import pixels  # pytype: disable=import-error
import numpy as np

from magi.agents import drq
from magi.agents.drq.agent import DrQConfig
from magi.agents.drq import networks
from magi.agents.drq import environment_loop
from magi.utils import loggers
from magi.utils.wrappers import FrameStackingWrapper
from magi.utils.wrappers import TakeKeyWrapper

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
flags.DEFINE_integer('action_repeat', None, '')
flags.DEFINE_integer('max_replay_size', 100_000, 'Minimum replay size')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('seed', 42, 'Random seed.')

PLANET_ACTION_REPEAT = {
    'cartpole-swingup': 8,
    'reacher-easy': 4,
    'cheetah-run': 4,
    'finger-spin': 2,
    'ball_in_cup-catch': 4,
    'walker-walk': 2
}


def load_env(domain_name, task_name, seed, frame_stack, action_repeat):
  env = suite.load(domain_name=domain_name,
                   task_name=task_name,
                   environment_kwargs={'flat_observation': True},
                   task_kwargs={'random': seed})
  env = pixels.Wrapper(env,
                       pixels_only=True,
                       render_kwargs={
                           'width': 84,
                           'height': 84,
                           'camera_id': 0
                       })
  env = wrappers.CanonicalSpecWrapper(env)
  env = TakeKeyWrapper(env, 'pixels')
  env = wrappers.ActionRepeatWrapper(env, action_repeat)
  env = FrameStackingWrapper(env, num_frames=frame_stack)
  env = wrappers.SinglePrecisionWrapper(env)
  return env


def main(_):
  import tensorflow as tf
  tf.config.set_visible_devices([], "GPU")
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
  if FLAGS.action_repeat is None:
    action_repeat = PLANET_ACTION_REPEAT.get(f'{FLAGS.domain_name}-{FLAGS.task_name}',
                                             None)
    if action_repeat is None:
      print('Unable to find action repeat configuration from PlaNet, default to 2')
      action_repeat = 2
  else:
    action_repeat = FLAGS.action_repeat
  env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed, FLAGS.frame_stack,
                 action_repeat)
  test_env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed + 42,
                      FLAGS.frame_stack, action_repeat)
  spec = specs.make_environment_spec(env)
  network_spec = networks.make_default_networks(spec)

  counter = counting.Counter()
  agent = drq.DrQAgent(environment_spec=spec,
                       networks=network_spec,
                       config=DrQConfig(
                           max_replay_size=FLAGS.max_replay_size,
                           batch_size=FLAGS.batch_size,
                           temperature_adam_b1=0.9,
                       ),
                       seed=FLAGS.seed,
                       logger=loggers.make_logger(label='learner',
                                                  log_frequency=1000,
                                                  use_wandb=FLAGS.wandb),
                       counter=counting.Counter(counter, 'learner'))
  eval_actor = agent.make_actor(is_eval=True)
  train_loop = environment_loop.ActionRepeatEnvironmentLoop(
      env,
      agent,
      logger=loggers.make_logger(label='train',
                                 log_frequency=5,
                                 use_wandb=FLAGS.wandb),
      counter=counting.Counter(counter, 'train'),
      action_repeat=action_repeat)
  eval_loop = environment_loop.ActionRepeatEnvironmentLoop(
      test_env,
      eval_actor,
      logger=loggers.make_logger(label='eval', use_wandb=FLAGS.wandb),
      counter=counting.Counter(counter, 'eval'),
      action_repeat=action_repeat)

  for _ in range(FLAGS.num_steps // (FLAGS.eval_freq * action_repeat)):
    train_loop.run(num_steps=FLAGS.eval_freq * action_repeat)
    eval_actor.update(wait=True)
    eval_loop.run(num_episodes=FLAGS.eval_episodes)

  if FLAGS.wandb:
    wandb.finish()


if __name__ == '__main__':
  app.run(main)
