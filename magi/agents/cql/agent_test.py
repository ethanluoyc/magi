"""Tests for CQL learner"""
from absl.testing import absltest
from acme import specs
from acme.testing import fakes
import jax
import optax

from magi.agents import sac
from magi.agents.cql import learning


class CQLTest(absltest.TestCase):
    def test_cql_learner_step(self):
        """Test that CQL learner step runs"""
        environment = fakes.ContinuousEnvironment(
            action_dim=2, observation_dim=2, bounded=True
        )
        environment_spec = specs.make_environment_spec(environment)
        agent_networks = sac.make_networks(
            environment_spec,
            policy_layer_sizes=(10, 10),
            critic_layer_sizes=(10, 10),
        )
        learner = learning.CQLLearner(
            policy_network=agent_networks["policy"],
            critic_network=agent_networks["critic"],
            random_key=jax.random.PRNGKey(0),
            dataset=fakes.transition_dataset(environment).batch(2).as_numpy_iterator(),
            policy_optimizer=optax.adam(1e-4),
            critic_optimizer=optax.adam(1e-4),
            alpha_optimizer=optax.adam(1e-4),
            target_entropy=sac.target_entropy_from_env_spec(environment_spec),
            init_alpha_prime=1.0,
            policy_eval_start=0,
            with_lagrange=True,
            lagrange_thresh=0.0,
            max_q_backup=False,
            deterministic_backup=True,
            num_random=2,
        )
        learner.step()


if __name__ == "__main__":
    absltest.main()
