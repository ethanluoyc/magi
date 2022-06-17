"""Tests for running DrQ agent."""
from absl.testing import absltest
import acme
from acme import specs
from acme.utils import loggers

from magi.agents import drq
from magi.agents import sac
from magi.testing import fakes


class DrQTest(absltest.TestCase):

  def test_drq(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousVisualEnvironment(
        action_dim=2,
        observation_shape=(32, 32, 3),
        episode_length=10,
        bounded=True)
    spec = specs.make_environment_spec(environment)
    # Construct the agent.
    agent_networks = drq.make_networks(
        spec,
        critic_hidden_sizes=(10, 10),
        actor_hidden_sizes=(10, 10),
        latent_size=10,
        num_layers=1,
        num_filters=2,
    )
    agent = drq.DrQAgent(
        environment_spec=spec,
        networks=agent_networks,
        seed=0,
        config=drq.DrQConfig(
            batch_size=2,
            target_entropy=sac.target_entropy_from_env_spec(spec),
            min_replay_size=1,
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
