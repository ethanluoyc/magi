"""Tests for running distribtued DrQ-v2 agent."""
from absl.testing import absltest
import acme
from acme import specs
from acme.utils import loggers

from magi.agents import drq_v2
from magi.testing import fakes


class DrQTest(absltest.TestCase):

  def test_drq_v2(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousVisualEnvironment(
        action_dim=2,
        observation_shape=(32, 32, 3),
        episode_length=10,
        bounded=True)
    spec = specs.make_environment_spec(environment)
    # Construct the agent.
    agent_networks = drq_v2.make_networks(spec, hidden_size=10, latent_size=10)
    agent = drq_v2.DrQV2(
        environment_spec=spec,
        networks=agent_networks,
        seed=0,
        config=drq_v2.DrQV2Config(
            batch_size=2,
            min_replay_size=10,
        ),
    )

    loop = acme.EnvironmentLoop(
        environment,
        agent,
        logger=loggers.make_default_logger(
            label='environment', save_data=False),
    )
    loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
