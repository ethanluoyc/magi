"""Tests for CRR learner."""
from absl.testing import absltest
from acme import specs
from acme.testing import fakes
import jax

from magi.agents import crr


class CRRLearnerTest(absltest.TestCase):
    def test_learner_step(self):
        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(
            action_dim=2, observation_dim=3, episode_length=10, bounded=True
        )
        spec = specs.make_environment_spec(environment)
        # # Try running the environment loop. We have no assertions here because all
        networks = crr.make_networks(spec, (10, 10), (10, 10))
        dataset = fakes.transition_dataset(environment).batch(10).as_numpy_iterator()
        learner = crr.CRRLearner(
            policy_network=networks["policy"],
            critic_network=networks["critic"],
            dataset=dataset,
            random_key=jax.random.PRNGKey(0),
        )
        learner.step()


if __name__ == "__main__":
    absltest.main()
