"""Running SAC on cartpole."""
import acme
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from acme import specs
from magi.agents.jax import sac
from magi.agents.jax.sac import acting
from dm_control import suite
from acme import specs
from magi.agents.jax.sac.networks import (ContinuousQFunction,
                                          StateDependentGaussianPolicy)
from magi.agents.jax.sac import loggers
from acme import wrappers
import wandb


_HIDDEN_UNITS = (1024, 1024)

def policy_fn(action_spec):

  def fn(x):
    return StateDependentGaussianPolicy(
        action_spec=action_spec,
        hidden_units=_HIDDEN_UNITS,
        log_std_min=-5,
        log_std_max=2,
    )(x["observations"])

  return fn


def critic_fn():

  def fn(x, a):
    critic = ContinuousQFunction(hidden_units=_HIDDEN_UNITS)
    return critic(x["observations"], a)

  return fn


def main(_):
  wandb.init(entity='ethanluoyc',
             project='magi',
             group='sac',
             name='cheetah_run_test')
  environment = suite.load("cheetah",
                           "run",
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
                       key=jax.random.PRNGKey(0),
                       logger=loggers.WandbLogger(label='learner', log_every_n_steps=1000))

  loop = acme.EnvironmentLoop(environment,
                              agent,
                              logger=loggers.WandbLogger(label='environment_loop'))
  loop.run(num_steps=int(1e7))


if __name__ == '__main__':
  app.run(main)