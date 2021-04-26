import acme
from acme.wrappers import action_repeat
import numpy as np
from absl import app, flags
from acme import specs, wrappers
from dm_control import suite
from magi.agents.sac_ae import networks
from magi.agents.sac_ae.agent import SACAEAgent
from magi.agents.jax.sac import loggers
from magi.agents.sac_ae2.sac_ae import SAC_AE
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
#   env = pixels.Wrapper(env, render_kwargs={'width': 84, 'height': 84})
#   env = wrappers.FrameStackingWrapper(env, num_frames=frame_stack)
#   env = wrappers.ActionRepeatWrapper(env, num_repeats=action_repeat)
#   env = ConcatFrameWrapper(env)
#   env = wrappers.CanonicalSpecWrapper(env)
#   env = wrappers.SinglePrecisionWrapper(env)
#   return env

def load_env(domain_name, task_name, seed, frame_stack, action_repeat):
  del frame_stack
  from magi.agents.sac_ae2.environment import make_dmc_env
  env = make_dmc_env(domain_name, task_name, action_repeat)
  env.seed(seed)
  from acme.wrappers import gym_wrapper
  return wrappers.SinglePrecisionWrapper(gym_wrapper.GymWrapper(env))

def evaluate(env, algo, num_eval_episodes):
  total_return = 0.0
  for _ in range(num_eval_episodes):
      timestep = env.reset()
      while not timestep.last():
        action = algo.greedy_select_action(timestep.observation)
        next_ts = env.step(action)
        total_return += next_ts.reward
        timestep = next_ts

  # Log mean return.
  mean_return = total_return / num_eval_episodes
  # To TensorBoard.
  # To CSV.
  # pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

  # Log to standard output.
  print(f"Return: {mean_return:<5.1f}")

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
  test_env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed + 1000, FLAGS.frame_stack, FLAGS.action_repeat)
  spec = specs.make_environment_spec(env)
  print(spec)

  algo = SAC_AE(environment_spec=spec, seed=FLAGS.seed)

  loop = acme.EnvironmentLoop(env,
                              algo,
                              logger=loggers.make_logger(label='environment_loop',
                                                         time_delta=5.,
                                                         use_wandb=FLAGS.wandb))
  # eval_loop = acme.EnvironmentLoop(env,
  #                                  algo.eval_actor,
  #                                  logger=loggers.make_logger(label='eval',
  #                                                             time_delta=0,
  #                                                             use_wandb=FLAGS.wandb))
  # loop.run(num_steps=FLAGS.num_steps)
  for _ in range(FLAGS.num_steps // FLAGS.eval_freq):
    loop.run(num_steps=FLAGS.eval_freq)
    # algo.eval_actor.update(wait=True)
    evaluate(test_env, algo, FLAGS.eval_episodes)
    # eval_loop.run(num_episodes=FLAGS.eval_episodes)

  if FLAGS.wandb:
    wandb.finish()


if __name__ == "__main__":
  app.run(main)
