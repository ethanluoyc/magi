"""Tests for CQL learner"""
from absl.testing import absltest
from acme import specs
from acme.testing import fakes
import jax
import optax

from magi.agents import cql
from magi.agents import sac


class CQLTest(absltest.TestCase):

  def test_cql_learner_step(self):
    """Test that CQL learner step runs"""
    environment = fakes.ContinuousEnvironment(
        action_dim=2, observation_dim=2, bounded=True)
    environment_spec = specs.make_environment_spec(environment)
    agent_networks = sac.make_networks(
        environment_spec,
        policy_layer_sizes=(10, 10),
        critic_layer_sizes=(10, 10),
    )
    learner = cql.CQLLearner(
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        random_key=jax.random.PRNGKey(0),
        dataset=fakes.transition_iterator(environment)(2),
        policy_optimizer=optax.adam(1e-4),
        critic_optimizer=optax.adam(1e-4),
        alpha_optimizer=optax.adam(1e-4),
        target_entropy=sac.target_entropy_from_env_spec(environment_spec),
        num_bc_steps=0,
        with_lagrange=True,
        target_action_gap=0.0,
        cql_alpha=5.0,
        max_q_backup=False,
        deterministic_backup=True,
        num_cql_samples=2,
    )
    learner.step()


if __name__ == '__main__':
  absltest.main()
