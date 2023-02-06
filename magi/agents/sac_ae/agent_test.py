"""Tests for running SAC-AE agent."""
import acme
from absl.testing import absltest
from acme import specs
from acme.utils import loggers

from magi.agents import sac_ae
from magi.testing import fakes


class SACAETest(absltest.TestCase):
    def test_sac_ae(self):
        # Create a fake environment to test with.
        environment = fakes.ContinuousVisualEnvironment(
            action_dim=2, observation_shape=(84, 84, 3), episode_length=10, bounded=True
        )
        spec = specs.make_environment_spec(environment)
        # Construct the agent.
        network_spec = sac_ae.make_default_networks(
            spec,
            actor_hidden_sizes=(32, 32),
            critic_hidden_sizes=(32, 32),
            num_layers=4,
        )
        agent = sac_ae.SACAEAgent(
            environment_spec=spec,
            networks=network_spec,
            seed=0,
            config=sac_ae.SACAEConfig(
                min_replay_size=1,
                initial_num_steps=0,
                batch_size=2,
            ),
        )

        loop = acme.EnvironmentLoop(
            environment,
            agent,
            logger=loggers.make_default_logger(label="environment", save_data=False),
        )
        loop.run(num_episodes=2)


if __name__ == "__main__":
    absltest.main()
