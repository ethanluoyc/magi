"""Running SAC on cartpole."""
import acme
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from acme import specs
from magi.agents.jax import sac
from dm_control import suite
from acme import specs
from magi.agents.jax.sac.networks import (ContinuousQFunction,
                                          StateDependentGaussianPolicy)
from acme import wrappers
from acme.utils import loggers


def policy_fn(action_spec):

  def fn(x):
    return StateDependentGaussianPolicy(action_spec=action_spec)(x["observations"])

  return fn


def critic_fn():

  def fn(x, a):
    critic = ContinuousQFunction()
    return critic(x["observations"], a)

  return fn


def main(_):
  # Create a fake environment to test with.
  environment = suite.load("cartpole",
                           "swingup",
                           environment_kwargs={"flat_observation": True})
  environment = wrappers.CanonicalSpecWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
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
  # we care about is that the agent runs without raising any errors.
  loop = acme.EnvironmentLoop(environment,
                              agent,
                              logger=loggers.make_default_logger(label='environment',
                                                                 save_data=False))
  loop.run(num_episodes=1000)


if __name__ == '__main__':
  app.run(main)
