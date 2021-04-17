"""Tests for soft actor critic."""
import acme
import haiku as hk
import jax
from absl.testing import absltest
from acme import specs
from acme.testing import fakes
from magi.agents.sac2 import networks
from magi.agents.sac2.agent import SACAgent
from acme.utils import loggers


def policy_fn(action_spec):

  def fn(x):
    return networks.GaussianPolicy(action_size=action_spec.shape[0])(x)

  return fn


def critic_fn():

  def fn(x, a):
    critic = networks.DoubleCritic()
    return critic(x, a)

  return fn


class SACTest(absltest.TestCase):

  def test_sac(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(action_dim=2,
                                              observation_dim=3,
                                              episode_length=10, bounded=True)
    spec = specs.make_environment_spec(environment)
    print(spec)

    # Make network purely functional
    policy = hk.without_apply_rng(hk.transform(policy_fn(spec.actions), apply_rng=True))
    critic = hk.without_apply_rng(hk.transform(critic_fn(), apply_rng=True))

    # Construct the agent.
    agent = SACAgent(environment_spec=spec,
                     policy=policy,
                     critic=critic,
                     seed=0,
                     start_steps=10,
                     batch_size=1)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment,
                                agent,
                                logger=loggers.make_default_logger(label='environment',
                                                                   save_data=False))
    loop.run(num_episodes=20)


if __name__ == '__main__':
  absltest.main()
