import time

import acme
import numpy as np
from absl import app, flags
from acme import specs, wrappers
from acme.wrappers import gym_wrapper
from magi.agents.jax.sac import loggers
from magi.agents import sac_ae2
from magi.agents.sac_ae2 import networks
from magi.agents.sac_ae2.environment import make_dmc_env

FLAGS = flags.FLAGS
flags.DEFINE_string('domain_name', 'cheetah', 'dm_control domain')
flags.DEFINE_string('task_name', 'run', 'dm_control task')
flags.DEFINE_bool('wandb', False, 'whether to log result to wandb')
flags.DEFINE_string('wandb_project', 'magi', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'ethanluoyc', 'wandb project entity')
flags.DEFINE_integer('num_steps', int(1e6), 'Random seed.')
flags.DEFINE_integer('eval_freq', 5000, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Random seed.')
flags.DEFINE_integer('frame_stack', 3, 'Random seed.')
flags.DEFINE_integer('action_repeat', 4, 'Random seed.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

# def load_env(domain_name, task_name, seed, frame_stack=3, action_repeat=4):
#   env = suite.load(domain_name=domain_name,
#                    task_name=task_name,
#                    environment_kwargs={"flat_observation": True},
#                    task_kwargs={'random': seed})
#   from dm_control.suite.wrappers import pixels
#   env = pixels.Wrapper(env, render_kwargs={'width': 84, 'height': 84}, pixels_only=True)
#   env = wrappers.ActionRepeatWrapper(env, num_repeats=action_repeat)
#   env = wrappers.FrameStackingWrapper(env, num_frames=frame_stack)
#   env = ConcatFrameWrapper(env)
#   env = wrappers.CanonicalSpecWrapper(env)
#   env = wrappers.SinglePrecisionWrapper(env)
#   return env


def load_env(domain_name, task_name, seed, frame_stack, action_repeat):
  # TODO(yl): Remove make_dmc_env and construct environment from dm_control directly.
  env = make_dmc_env(domain_name, task_name, action_repeat, n_frames=frame_stack, image_size=84)
  env.seed(seed)
  return wrappers.SinglePrecisionWrapper(gym_wrapper.GymWrapper(env))


def main(_):
  np.random.seed(FLAGS.seed)
  if FLAGS.wandb:
    import wandb
    wandb.init(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        name=
        f"sac_ae-{FLAGS.domain_name}-{FLAGS.task_name}_{FLAGS.seed}_{int(time.time())}",
        config=FLAGS)
  env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed, FLAGS.frame_stack,
                 FLAGS.action_repeat)
  test_env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed + 1000,
                      FLAGS.frame_stack, FLAGS.action_repeat)
  spec = specs.make_environment_spec(env)
  network_spec = networks.make_default_networks(spec)

  agent = sac_ae2.SACAEAgent(environment_spec=spec,
                     networks=network_spec,
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


if __name__ == "__main__":
  app.run(main)
