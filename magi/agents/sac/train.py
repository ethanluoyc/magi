import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from dm_control import suite  # pytype: disable=import-error
import haiku as hk
import numpy as np

from magi.agents.sac.agent import SACAgent
from magi.agents.sac import networks
from magi.utils import loggers

FLAGS = flags.FLAGS
flags.DEFINE_string('domain_name', 'cartpole', 'dm_control domain')
flags.DEFINE_string('task_name', 'swingup', 'dm_control task')
flags.DEFINE_bool('wandb', False, 'whether to log result to wandb')
flags.DEFINE_string('wandb_project', 'magi', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'ethanluoyc', 'wandb project entity')
flags.DEFINE_integer('num_steps', int(1e6), 'Random seed.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def load_env(domain_name, task_name, seed):
  env = suite.load(domain_name=domain_name,
                   task_name=task_name,
                   environment_kwargs={'flat_observation': True},
                   task_kwargs={'random': seed})
  env = wrappers.CanonicalSpecWrapper(env)
  env = wrappers.SinglePrecisionWrapper(env)
  return env


def main(_):
  np.random.seed(FLAGS.seed)
  if FLAGS.wandb:
    import wandb  # pylint: disable=import-outside-toplevel
    wandb.init(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        name=f'{FLAGS.domain_name}-{FLAGS.task_name}_{FLAGS.seed}_{int(time.time())}',
        config=FLAGS)
  env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed)
  spec = specs.make_environment_spec(env)

  def critic_fn(s, a):
    return networks.DoubleCritic(hidden_units=(256, 256),)(s['observations'], a)

  def policy_fn(s):
    return networks.GaussianPolicy(
        action_size=spec.actions.shape[0],
        hidden_units=(256, 256),
    )(s['observations'])

  policy = hk.without_apply_rng(hk.transform(policy_fn, apply_rng=True))
  critic = hk.without_apply_rng(hk.transform(critic_fn, apply_rng=True))

  algo = SACAgent(environment_spec=spec,
                  policy=policy,
                  critic=critic,
                  seed=FLAGS.seed,
                  logger=loggers.make_logger(label='learner',
                                             log_frequency=1000,
                                             use_wandb=FLAGS.wandb))

  loop = acme.EnvironmentLoop(env,
                              algo,
                              logger=loggers.make_logger(label='environment_loop',
                                                         use_wandb=FLAGS.wandb))
  loop.run(num_steps=FLAGS.num_steps)
  if FLAGS.wandb:
    wandb.finish()


if __name__ == '__main__':
  app.run(main)
