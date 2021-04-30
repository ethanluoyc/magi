"""Tests for PPO agent."""
from absl.testing import absltest
import acme
from acme import specs
from acme.testing import fakes
import haiku as hk
import jax
import numpy as np

from magi.agents import ppo


def make_network(action_spec):
    def network(obs):
        torso = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(64),
                jax.nn.relu,
                hk.Linear(64),
                jax.nn.relu,
            ]
        )
        embedding = torso(obs)
        return (
            hk.Linear(action_spec.num_values)(embedding),
            hk.Linear(1)(embedding).squeeze(-1),
        )

    return network


class PPOTest(absltest.TestCase):
    def test_ppo(self):
        # Create a fake environment to test with.
        environment = fakes.DiscreteEnvironment(
            num_actions=5,
            num_observations=10,
            action_dtype=np.int64,
            obs_shape=(10, 5),
            obs_dtype=np.float32,
            episode_length=9,
        )
        spec = specs.make_environment_spec(environment)

        agent = ppo.PPO(
            environment_spec=specs.make_environment_spec(environment),
            network_fn=make_network(spec.actions),
            sequence_length=10,
            sequence_period=10,
            seed=0,
        )

        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(environment, agent)
        loop.run(num_episodes=20)


if __name__ == "__main__":
    absltest.main()
