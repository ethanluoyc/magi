from typing import Optional
import haiku as hk
import jax.numpy as jnp
from acme import core
import dm_env
import numpy as np
import jax

import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

class SACActor(core.Actor):
  """A SAC actor."""

  def __init__(
      self,
      forward_fn,
      rng,
      variable_client,
      adder = None,
  ):

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._key = next(rng)

    @jax.jit
    def forward(params, key, observation):
      key, subkey = jax.random.split(key)
      mean, logstd = forward_fn(params, observation)
      action_base_dist = tfd.Normal(
          loc=mean,
          scale=jnp.exp(logstd)
      )
      action_dist = tfd.TransformedDistribution(
          action_base_dist, tfb.Tanh()
      )
      action = action_dist.sample(seed=subkey)
      return action, key

    self._forward = forward
    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    if self._variable_client is not None:
      self._variable_client.update_and_wait()

  def select_action(self, observation):
    # Forward.
    action, self._key = self._forward(self._params, self._key, observation)
    action = np.array(action)
    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder is not None:
      self._adder.add_first(timestep)

  def observe(
      self,
      action,
      next_timestep: dm_env.TimeStep,
  ):
    if self._adder is not None:
      self._adder.add(action, next_timestep)

  def update(self, wait: bool = True): # not the default wait = False
    if self._variable_client is not None:
      self._variable_client.update_and_wait()

  @property
  def _params(self) -> Optional[hk.Params]:
    if self._variable_client is None:
      # If self._variable_client is None then we assume self._forward  does not
      # use the parameters it is passed and just return None.
      return None
    return self._variable_client.params


class RandomActor(core.Actor):
  """A SAC actor."""

  def __init__(
      self,
      action_spec,
      rng,
      adder = None,
  ):

    # Store these for later use.
    self._adder = adder
    self._variable_client = None
    self._rng = rng
    self._action_spec = action_spec

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    if self._variable_client is not None:
      self._variable_client.update_and_wait()


  def select_action(self, observation):
    # Forward.
    del observation
    action_dist = tfd.Uniform(low=self._action_spec.minimum, high=self._action_spec.maximum)
    action = action_dist.sample(seed=next(self._rng))
    action = np.array(action)
    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder is not None:
      self._adder.add_first(timestep)

  def observe(
      self,
      action,
      next_timestep: dm_env.TimeStep,
  ):
    if self._adder is not None:
      self._adder.add(action, next_timestep)

  def update(self, wait: bool = True): # not the default wait = False
    if self._variable_client is not None:
      self._variable_client.update(wait)

  @property
  def _params(self) -> Optional[hk.Params]:
    if self._variable_client is None:
      # If self._variable_client is None then we assume self._forward  does not
      # use the parameters it is passed and just return None.
      return None
    return self._variable_client.params

