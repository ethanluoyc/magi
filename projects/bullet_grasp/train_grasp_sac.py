import time

from absl import app
from absl import flags
import acme
from acme.jax.networks import distributional
from acme import specs
from acme.utils import counting
from acme.wrappers import gym_wrapper
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from magi.agents.sac.agent import SACAgent
from magi.experimental.environments import bullet_kuka_env
from magi.utils import loggers
import utils

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
    h = hk.Flatten()(h)
    output = jax.nn.relu(
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h))
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

  agent = SACAgent(environment_spec=spec,
                   policy=policy,
                   critic=critic,
                   seed=FLAGS.seed,
                   logger=loggers.make_logger(label='learner',
                                              log_frequency=500,
                                              use_wandb=FLAGS.wandb),
                   initial_num_steps=FLAGS.min_num_steps,
                   batch_size=FLAGS.batch_size)
  eval_actor = agent.make_actor(is_eval=False)
  counter = counting.Counter()
  loop = acme.EnvironmentLoop(env,
                              agent,
                              logger=loggers.make_logger(label='environment_loop',
                                                         use_wandb=FLAGS.wandb),
                              counter=counter)
  eval_logger = loggers.make_logger(label='evaluation', use_wandb=FLAGS.wandb)
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    loop.run(num_steps=FLAGS.eval_every)
    eval_stats = utils.evaluate(eval_actor, env)
    eval_logger.write({**eval_stats, 'steps': counter.get_counts()['steps']})
  if FLAGS.wandb:
    wandb.finish()


if __name__ == '__main__':
  app.run(main)
