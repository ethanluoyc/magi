"""Tests for TD3-BC."""
from absl.testing import absltest
from acme import specs
from acme.testing import fakes
import jax

from magi.agents import td3
from magi.agents import td3_bc


class TD3BCTest(absltest.TestCase):

  def test_td3(self):
    environment = fakes.ContinuousEnvironment(
        action_dim=2, observation_dim=3, episode_length=10, bounded=True)
    spec = specs.make_environment_spec(environment)
    # # Try running the environment loop. We have no assertions here because all
    agent_networks = td3.make_networks(spec, (10,), (10,))
    dataset = fakes.transition_dataset(environment).batch(
        10).as_numpy_iterator()
    learner = td3_bc.TD3BCLearner(
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        random_key=jax.random.PRNGKey(0),
        iterator=dataset,
    )
    learner.step()


if __name__ == '__main__':
  absltest.main()
