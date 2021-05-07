"""Tests for running SAC-AE agent."""
from absl.testing import absltest
import acme
from acme import specs
from acme.utils import loggers

from magi.agents.drq.agent import DrQAgent
from magi.agents.drq.agent import DrQConfig
from magi.agents.drq import networks
from magi.utils import fakes


class SACTest(absltest.TestCase):

  def test_sac_ae(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousVisualEnvironment(action_dim=2,
                                                    observation_shape=(84, 84, 3),
                                                    episode_length=10,
                                                    bounded=True)
    spec = specs.make_environment_spec(environment)
    # Construct the agent.
    network_spec = networks.make_default_networks(spec)
    agent = DrQAgent(environment_spec=spec,
                     networks=network_spec,
                     seed=0,
                     config=DrQConfig(initial_num_steps=10, batch_size=2))

    loop = acme.EnvironmentLoop(environment,
                                agent,
                                logger=loggers.make_default_logger(label='environment',
                                                                   save_data=False))
    loop.run(num_episodes=20)


if __name__ == '__main__':
  absltest.main()
