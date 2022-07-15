from absl.testing import absltest
import acme
from acme import types
import itertools
from acme.jax import utils
from acme.testing import fakes
import optax
import jax
import jax.numpy as jnp
import reverb
import tensorflow_probability.substrates.jax as tfp

from magi.agents.rhpo import networks
from magi.agents.rhpo import learning

tfd = tfp.distributions


class RHPOTest(absltest.TestCase):

  def test_rhpo_learner(self):
    obs_size = 3
    act_size = 2
    num_tasks = 2
    num_components = 3

    environment = fakes.ContinuousEnvironment(
        observation_dim=obs_size, action_dim=act_size)

    environment_spec = acme.make_environment_spec(environment)
    agent_networks = networks.make_networks(
        environment_spec, num_tasks=num_tasks, num_components=num_components)
    policy_network = agent_networks['policy']
    critic_network = agent_networks['critic']
    key = jax.random.PRNGKey(0)

    dummy_obs = utils.add_batch_dim(
        utils.zeros_like(environment_spec.observations))
    dummy_act = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))

    transition = types.Transition(
        observation=dummy_obs,
        next_observation=dummy_obs,
        action=dummy_act,
        reward=jnp.ones((1, num_tasks)),
        discount=jnp.ones((1,)),
    )
    iterator = (
        reverb.ReplaySample(None, data=transition) for _ in itertools.count())
    learner = learning.RHPOLearner(
        policy_network,
        critic_network,
        iterator,
        key,
        optax.adam(1e-4),
        optax.adam(1e-4),
        optax.adam(1e-4),
        num_tasks=num_tasks,
        num_components=num_components,
        discount=0.99,
        num_samples=10,
        action_dim=act_size,
        target_policy_update_period=100,
        target_critic_update_period=100,
    )
    learner.step()

if __name__ == '__main__':
  absltest.main()