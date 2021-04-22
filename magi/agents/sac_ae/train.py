import acme
from acme.wrappers import action_repeat
import numpy as np
from absl import app, flags
from acme import specs, wrappers
from dm_control import suite
from magi.agents.sac_ae import networks
from magi.agents.sac_ae.agent import SACAEAgent
from magi.agents.jax.sac import loggers
from magi.agents.sac_ae.wrappers import ConcatFrameWrapper
import haiku as hk
import time

FLAGS = flags.FLAGS
flags.DEFINE_string('domain_name', 'cheetah', 'dm_control domain')
flags.DEFINE_string('task_name', 'run', 'dm_control task')
flags.DEFINE_bool('wandb', False, 'whether to log result to wandb')
flags.DEFINE_string('wandb_project', 'magi', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'ethanluoyc', 'wandb project entity')
flags.DEFINE_integer('num_steps', int(1e6), 'Random seed.')
flags.DEFINE_integer('eval_freq', 10000, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Random seed.')
flags.DEFINE_integer('frame_stack', 3, 'Random seed.')
flags.DEFINE_integer('action_repeat', 4, 'Random seed.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def load_env(domain_name, task_name, seed, frame_stack=3, action_repeat=4):
  env = suite.load(domain_name=domain_name,
                   task_name=task_name,
                   environment_kwargs={"flat_observation": True},
                   task_kwargs={'random': seed})
  from dm_control.suite.wrappers import pixels
  env = pixels.Wrapper(env, render_kwargs={'width': 84, 'height': 84})
  env = wrappers.FrameStackingWrapper(env, num_frames=frame_stack)
  env = wrappers.ActionRepeatWrapper(env, action_repeat)
  env = ConcatFrameWrapper(env)
  env = wrappers.CanonicalSpecWrapper(env)
  env = wrappers.SinglePrecisionWrapper(env)
  return env


def main(_):
  np.random.seed(FLAGS.seed)
  if FLAGS.wandb:
    import wandb
    wandb.init(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        name=f"sac_ae-{FLAGS.domain_name}-{FLAGS.task_name}_{FLAGS.seed}_{int(time.time())}",
        config=FLAGS)
  env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed, FLAGS.frame_stack, FLAGS.action_repeat)
  spec = specs.make_environment_spec(env)
  print(spec)

  policy = hk.without_apply_rng(
      hk.transform(lambda f: networks.Policy(spec.actions.shape[0])(f), apply_rng=True))
  critic = hk.without_apply_rng(
      hk.transform(lambda o, a: networks.Critic()(o, a), apply_rng=True))
  encoder = hk.without_apply_rng(
      hk.transform(lambda o: networks.SACEncoder()(o['pixels'])))
  decoder = hk.without_apply_rng(
      hk.transform(lambda f: networks.SACDecoder(spec.observations['pixels'].shape[-1])
                   (f)))
  linear = hk.without_apply_rng(
      hk.transform(lambda f: networks.SACLinear(50)(f)))

  algo = SACAEAgent(environment_spec=spec,
                    policy=policy,
                    critic=critic,
                    linear=linear,
                    encoder=encoder,
                    decoder=decoder,
                    seed=FLAGS.seed,
                    logger=loggers.make_logger(label='learner',
                                               time_delta=30,
                                               use_wandb=FLAGS.wandb),
                    batch_size=128, start_steps=1000, 
                    init_alpha=0.1)

  loop = acme.EnvironmentLoop(env,
                              algo,
                              logger=loggers.make_logger(label='environment_loop',
                                                         time_delta=5.,
                                                         use_wandb=FLAGS.wandb))
  eval_loop = acme.EnvironmentLoop(env,
                                   algo.eval_actor,
                                   logger=loggers.make_logger(label='eval',
                                                              time_delta=0,
                                                              use_wandb=FLAGS.wandb))
  for _ in range(FLAGS.num_steps // FLAGS.eval_freq):
    loop.run(num_steps=FLAGS.eval_freq)
    algo.eval_actor.update(wait=True)
    eval_loop.run(num_episodes=FLAGS.eval_episodes)

  if FLAGS.wandb:
    wandb.finish()


if __name__ == "__main__":
  app.run(main)
