import acme
import numpy as np
from absl import app, flags
from acme import specs, wrappers
from dm_control import suite
from magi.agents.sac2 import networks
from magi.agents.sac2.agent import SACAgent
import haiku as hk

FLAGS = flags.FLAGS
flags.DEFINE_string('domain_name', 'cartpole', 'dm_control domain')
flags.DEFINE_string('task_name', 'swingup', 'dm_control task')
flags.DEFINE_integer('num_steps', int(1e6), 'Random seed.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def load_env(domain_name, task_name, seed):
  env = suite.load(domain_name=domain_name,
                   task_name=task_name,
                   environment_kwargs={"flat_observation": True},
                   task_kwargs={'random': seed})
  env = wrappers.CanonicalSpecWrapper(env)
  env = wrappers.SinglePrecisionWrapper(env)
  return env


def main(_):
  np.random.seed(FLAGS.seed)
  env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed)
  spec = specs.make_environment_spec(env)

  def critic_fn(s, a):
    return networks.DoubleCritic(hidden_units=(256, 256),)(s['observations'], a)

  def policy_fn(s):
    return networks.GaussianPolicy(
        action_size=spec.actions.shape[0],
        hidden_units=(256, 256),
        log_std_min=-20,
        log_std_max=2,
    )(s['observations'])

  policy = hk.without_apply_rng(hk.transform(policy_fn, apply_rng=True))
  critic = hk.without_apply_rng(hk.transform(critic_fn, apply_rng=True))

  algo = SACAgent(
      environment_spec=spec,
      policy=policy,
      critic=critic,
      seed=FLAGS.seed,
  )

  loop = acme.EnvironmentLoop(env, algo)
  loop.run(num_steps=FLAGS.num_steps)


if __name__ == "__main__":
  app.run(main)
