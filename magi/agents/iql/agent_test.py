"""Integration tests for IQL learner"""
from absl.testing import absltest
from acme import specs
from acme.testing import fakes
import jax
import optax

from magi.agents import iql


class IQLAgentTest(absltest.TestCase):
    def test_agent_run(self):
        environment = fakes.ContinuousEnvironment(
            action_dim=2, observation_dim=3, episode_length=10, bounded=True
        )
        spec = specs.make_environment_spec(environment)
        # # Try running the environment loop. We have no assertions here because all
        agent_networks = iql.make_networks(spec, (10, 10), dropout_rate=0.1)
        dataset = fakes.transition_iterator(environment)(2)
        learner = iql.IQLLearner(
            random_key=jax.random.PRNGKey(0),
            networks=agent_networks,
            dataset=dataset,
            policy_optimizer=optax.adam(1e-4),
            critic_optimizer=optax.adam(1e-4),
            value_optimizer=optax.adam(1e-4),
        )
        learner.step()


if __name__ == "__main__":
    absltest.main()
