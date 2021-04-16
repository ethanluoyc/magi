"""Tests for soft actor critic."""
import acme
import haiku as hk
import jax
from absl.testing import absltest
from acme import specs
from acme.testing import fakes
from magi.agents.jax import sac
from magi.agents.jax.sac.networks import (ContinuousQFunction,
                                          StateDependentGaussianPolicy)


def policy_fn(action_spec):

  def fn(x):
    return StateDependentGaussianPolicy(action_spec=action_spec)(x)

  return fn


def critic_fn():

  def fn(x, a):
    critic = ContinuousQFunction()
    return critic(x, a)
  return fn


class SACTest(absltest.TestCase):

  def test_sac(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(action_dim=2,
                                              observation_dim=3,
                                              episode_length=10)
    spec = specs.make_environment_spec(environment)

    # Make network purely functional
    policy = hk.without_apply_rng(hk.transform(policy_fn(spec.actions), apply_rng=True))
    critic = hk.without_apply_rng(hk.transform(critic_fn(), apply_rng=True))

    # Construct the agent.
    agent = sac.SACAgent(environment_spec=spec,
                         policy_network=policy,
                         critic_network=critic,
                         key=jax.random.PRNGKey(0))

    # Try running the environment loop. We have no assertions here because all
    from acme.utils import loggers
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent, logger=loggers.make_default_logger(label='environment', save_data=False))
    loop.run(num_episodes=200)


#   @hk.testing.transform_and_run
#   def testPolicy(self):
#     policy = StateDependentGaussianPolicy(action_space=jnp.zeros(2))
#     mean, logstd = policy(jnp.zeros((1, 5)))
#     self.assertEqual(mean.shape, (1, 2))
#     self.assertEqual(logstd.shape, (1, 2))

#   @hk.testing.transform_and_run
#   def testCritic(self):
#     critic = ContinuousQFunction(num_critics=2)
#     batch_size = 7
#     state = jnp.zeros((7, 3))
#     action = jnp.zeros((7, 2))
#     q_values = critic(state, action)
#     self.assertEqual(len(q_values), 2)
#     self.assertEqual(q_values[0].shape, (7, 1))

#   @hk.testing.transform_and_run
#   def testComputeTarget(self):
#     # Randomly sample a batch of transitions
#     batch_size = 32
#     state_size = 5
#     action_size = 2
#     o_t = jnp.zeros((batch_size, state_size))
#     o_tp1 = jnp.ones((batch_size, state_size))
#     a_t = jnp.zeros((batch_size, action_size))
#     r_t = jnp.zeros((batch_size, 1))
#     d_t = jnp.zeros((batch_size, 1))
#     discount = 0.9
#     alpha = .5
#     policy = StateDependentGaussianPolicy(action_space=jnp.zeros(2))
#     # Compute targets for the Q functions
#     # y(r, s', d)
#     target = r_t + discount * (1 - d_t) # (alpha * )

# Update Q-function by one step SGD
# Update policy by one step of gradient descent
# Update target networks

if __name__ == '__main__':
  absltest.main()
