"""Integration tests for IQL learner"""
from absl.testing import absltest
import jax.numpy as jnp

from magi.agents.iql import networks
from magi.agents.iql import types
from magi.agents.iql.learning import Learner


class IQLAgentTest(absltest.TestCase):
    def test_agent_run(self):
        observation_size = 3
        action_size = 2
        agent_networks = networks.make_networks(
            jnp.zeros((observation_size,)),
            jnp.zeros((action_size,)),
            hidden_dims=(10, 10),
        )
        agent = Learner(0, agent_networks, max_steps=100)
        batch_size = 3
        batch = types.Batch(
            observations=jnp.zeros((batch_size, observation_size)),
            actions=jnp.zeros((batch_size, action_size)),
            rewards=jnp.zeros((batch_size,)),
            masks=jnp.zeros((batch_size,)),
            next_observations=jnp.zeros((batch_size, observation_size)),
        )
        agent.update(batch)


if __name__ == "__main__":
    absltest.main()
