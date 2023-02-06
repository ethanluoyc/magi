"""Tests for SAC-v1 agent."""
import jax
import optax
from absl.testing import absltest
from acme import specs
from acme.testing import fakes

from magi.agents import sac
from magi.agents import sac_v1


class SACV1Test(absltest.TestCase):
    def test_sac_v1(self):
        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(
            action_dim=2, observation_dim=3, episode_length=10, bounded=True
        )
        spec = specs.make_environment_spec(environment)

        hidden_layer_sizes = (10,)
        agent_networks = sac_v1.make_networks(
            spec,
            policy_layer_sizes=hidden_layer_sizes,
            critic_layer_sizes=hidden_layer_sizes,
            value_layer_sizes=hidden_layer_sizes,
        )

        # Test that learner runs
        rng_key = jax.random.PRNGKey(0)
        learner = sac_v1.SACV1Learner(
            agent_networks["policy"],
            agent_networks["critic"],
            agent_networks["value"],
            random_key=rng_key,
            dataset=fakes.transition_dataset(environment).batch(2).as_numpy_iterator(),
            policy_optimizer=optax.adam(3e-4),
            critic_optimizer=optax.adam(3e-4),
            value_optimizer=optax.adam(3e-4),
            alpha_optimizer=optax.adam(3e-4),
            target_entropy=sac.target_entropy_from_env_spec(spec),
        )
        learner.step()

    def test_raises_on_setting_target_with_fixed_coef(self):
        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(
            action_dim=2, observation_dim=3, episode_length=10, bounded=True
        )
        spec = specs.make_environment_spec(environment)

        hidden_layer_sizes = (10,)
        agent_networks = sac_v1.make_networks(
            spec,
            policy_layer_sizes=hidden_layer_sizes,
            critic_layer_sizes=hidden_layer_sizes,
            value_layer_sizes=hidden_layer_sizes,
        )

        # Test that learner runs
        rng_key = jax.random.PRNGKey(0)
        with self.assertRaises(ValueError):
            dataset = fakes.transition_dataset(environment).batch(2).as_numpy_iterator()
            learner = sac_v1.SACV1Learner(
                agent_networks["policy"],
                agent_networks["critic"],
                agent_networks["value"],
                random_key=rng_key,
                dataset=dataset,
                policy_optimizer=optax.adam(3e-4),
                critic_optimizer=optax.adam(3e-4),
                value_optimizer=optax.adam(3e-4),
                entropy_coefficient=1.0,
                target_entropy=-1.0,
            )
            learner.step()


if __name__ == "__main__":
    absltest.main()
