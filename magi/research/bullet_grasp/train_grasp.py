from typing import Sequence
import acme
import numpy as np
from acme.utils import counting
from absl import app, flags
from acme import specs
from magi.agents.sac2.agent import SACAgent
from magi.agents.jax.sac import loggers
import haiku as hk
import time
from acme.jax.networks import distributional
from acme.wrappers import gym_wrapper
from magi.experimental.environments import bullet_kuka_env
import jax.numpy as jnp
import jax

FLAGS = flags.FLAGS
flags.DEFINE_bool('wandb', False, 'whether to log result to wandb')
flags.DEFINE_string('name', 'kuka_grasp', 'experiment name')
flags.DEFINE_string('wandb_project', 'magi', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'ethanluoyc', 'wandb project entity')
flags.DEFINE_string('wandb_group', '', 'wandb project entity')
flags.DEFINE_integer('num_steps', int(5e5), 'Random seed.')
flags.DEFINE_integer('min_num_steps', int(1e3), 'Random seed.')
flags.DEFINE_integer('eval_every', int(1000), 'Random seed.')
flags.DEFINE_integer('batch_size', 128, 'Random seed.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


class GraspNetTorso(hk.Module):

  def __call__(self, observation):
    assert len(observation.shape) in [3, 4]
    expand_obs = len(observation.shape) == 3
    if expand_obs:
      observation = jnp.expand_dims(observation, 0)
    observation = observation.astype(jnp.float32) / 255.
    h = jax.nn.relu(hk.Conv2D(32, (3, 3), stride=2)(observation))
    h = jax.nn.relu(hk.Conv2D(32, (3, 3), stride=2)(h))
    h = hk.Conv2D(32, (3, 3), stride=1)(h)
    h = jax.nn.relu(hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h))
    output = hk.Flatten()(h)
    if expand_obs:
      output = jnp.squeeze(output, 0)
    return output


class GaussianPolicy(hk.Module):
  """
    Policy for SAC.
    """

  def __init__(self, action_size, name=None):
    super().__init__(name=name)
    self.action_size = action_size

  def __call__(self, x):
    torso = GraspNetTorso()
    h = torso(x)
    h = jax.nn.relu(hk.Linear(256)(h))
    return distributional.NormalTanhDistribution(self.action_size)(h)


class DoubleCritic(hk.Module):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.num_critics = 2

  def __call__(self, s, a):

    def _fn(x, a):
      torso = GraspNetTorso()
      embedding = jnp.concatenate([torso(x), a], -1)
      head = hk.Sequential([
          hk.Linear(256),
          jax.nn.relu,
          hk.Linear(1),
      ])
      return head(embedding).squeeze(-1)

    # Return list even if num_critics == 1 for simple implementation.
    return [_fn(s, a) for _ in range(self.num_critics)]


def load_env(seed):
  # Update this if necessary for change the environment
  env = bullet_kuka_env.KukaDiverseObjectEnv(renders=False,
                                             isDiscrete=False,
                                             width=64,
                                             height=64,
                                             numObjects=1,
                                             maxSteps=8,
                                             blockRandom=0,
                                             cameraRandom=0)
  env.seed(seed)
  env = gym_wrapper.GymWrapper(env)
  return env

def evaluate(actor, env, num_episodes=200):
  actor.update(wait=True)
  episode_lengths = []
  episode_returns = []

  for i in range(num_episodes):
    ep_step = 0
    ep_ret = 0
    timestep = env.reset()
    actor.observe_first(actor)
    while not timestep.last():
      if i == 0:
        action = actor.select_action(timestep.observation)
      timestep = env.step(action)
      ep_step += 1
      ep_ret += timestep.reward
    episode_lengths.append(ep_step)
    episode_returns.append(ep_ret)
  # all_probs = np.asarray(jnp.stack(all_probs, axis=-1))
  # fig, ax = plt.subplots(figsize=(12, 2))
  # ax.pcolormesh(all_probs[:, :162], cmap="Blues")
  # plt.close(fig)
  return {
      "eval_average_episode_length": np.mean(episode_lengths),
      "eval_average_episode_return": np.mean(episode_returns),
  }

def main(_):
  np.random.seed(FLAGS.seed)
  if FLAGS.wandb:
    import wandb
    wandb.init(project=FLAGS.wandb_project,
               entity=FLAGS.wandb_entity,
               name=f"{FLAGS.name}_{time.time()}",
               group=FLAGS.wandb_group,
               config=FLAGS)
  env = load_env(FLAGS.seed)
  spec = specs.make_environment_spec(env)

  def critic_fn(s, a):
    return DoubleCritic()(s, a)

  def policy_fn(s):
    return GaussianPolicy(action_size=spec.actions.shape[0],)(s)

  policy = hk.without_apply_rng(hk.transform(policy_fn, apply_rng=True))
  critic = hk.without_apply_rng(hk.transform(critic_fn, apply_rng=True))

  algo = SACAgent(environment_spec=spec,
                  policy=policy,
                  critic=critic,
                  seed=FLAGS.seed,
                  logger=loggers.make_logger(label='learner',
                                             time_delta=5.,
                                             use_wandb=FLAGS.wandb),
                  start_steps=FLAGS.min_num_steps,
                  batch_size=FLAGS.batch_size)
  counter = counting.Counter()
  loop = acme.EnvironmentLoop(env,
                              algo,
                              logger=loggers.make_logger(label='environment_loop',
                                                         time_delta=5.,
                                                         use_wandb=FLAGS.wandb), counter=counter)
  eval_logger = loggers.make_logger(label='evaluation',
                                    time_delta=0,
                                    use_wandb=FLAGS.wandb)
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    loop.run(num_steps=FLAGS.eval_every)
    eval_stats = evaluate(algo._eval_actor, env)
    eval_logger.write({**eval_stats, 'steps': counter.get_counts()['steps']})
  if FLAGS.wandb:
    wandb.finish()


if __name__ == "__main__":
  app.run(main)
