"""Tests for soft actor critic."""
from absl.testing import absltest
import acme
from acme import specs
from acme.testing import fakes
from acme.utils import loggers

from magi.agents.sac import config as sac_config
from magi.agents.sac import networks
from magi.agents.sac.agent import SACAgent


class SACTest(absltest.TestCase):
    def test_sac(self):
        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(
            action_dim=2, observation_dim=3, episode_length=10, bounded=True
        )
        spec = specs.make_environment_spec(environment)

        # Make network purely functional
        agent_networks = networks.make_networks(
            spec,
            policy_layer_sizes=(32, 32),
            critic_layer_sizes=(32, 32),
        )

        # Construct the agent.
        agent = SACAgent(
            environment_spec=spec,
            networks=agent_networks,
            config=sac_config.SACConfig(
                target_entropy=sac_config.target_entropy_from_env_spec(spec),
                min_replay_size=1,
                batch_size=1,
            ),
            seed=0,
        )

        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(
            environment,
            agent,
            logger=loggers.make_default_logger(label="environment", save_data=False),
        )
        loop.run(num_episodes=20)


if __name__ == "__main__":
    absltest.main()
