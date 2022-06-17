"""Default networks for CQL."""
from typing import Sequence

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils
from acme.jax.networks import distributional
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability

hk_init = hk.initializers
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

Initializer = hk.initializers.Initializer


class NormalTanhDistribution(hk.Module):
  """Module that produces a TanhTransformedDistribution distribution.
    A fork of the acme.jax.networks.NormalTanhDistribution that uses
    exp transform and clipping the maximum of the log_std
    """

  def __init__(
      self,
      num_dimensions: int,
      w_init: hk_init.Initializer = hk_init.VarianceScaling(
          1.0, 'fan_in', 'uniform'),
      b_init: hk_init.Initializer = hk_init.Constant(0.0),
  ):
    """Initialization.

      Args:
        num_dimensions: Number of dimensions of a distribution.
        min_scale: Minimum standard deviation.
        w_init: Initialization for linear layer weights.
        b_init: Initialization for linear layer biases.
    """
    super().__init__(name='Normal')
    self._loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
    self._scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    loc = self._loc_layer(inputs)
    scale = self._scale_layer(inputs)
    scale = jnp.exp(jnp.clip(scale, -20.0, 2.0))
    distribution = tfd.Normal(loc=loc, scale=scale)
    return tfd.Independent(
        distributional.TanhTransformedDistribution(distribution),
        reinterpreted_batch_ndims=1,
    )


def make_default_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256),
    critic_layer_sizes: Sequence[int] = (256, 256),
):
  num_dimensions = np.prod(spec.actions.shape, dtype=int)

  def _actor_fn(obs):
    network = hk.Sequential([
        hk.nets.MLP(
            list(policy_layer_sizes),
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=jax.nn.relu,
            activate_final=True,
        ),
        NormalTanhDistribution(num_dimensions),
    ])
    return network(obs)

  def _critic_fn(obs, action):
    network1 = hk.nets.MLP(
        list(critic_layer_sizes) + [1],
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
        activation=jax.nn.relu,
        activate_final=False,
    )
    network2 = hk.nets.MLP(
        list(critic_layer_sizes) + [1],
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
        activation=jax.nn.relu,
        activate_final=False,
    )
    input_ = jnp.concatenate([obs, action], axis=-1)
    value1 = network1(input_)
    value2 = network2(input_)
    return jnp.squeeze(value1, axis=-1), jnp.squeeze(value2, axis=-1)

  dummy_action = jax_utils.zeros_like(spec.actions)
  dummy_obs = jax_utils.zeros_like(spec.observations)
  dummy_action = jax_utils.add_batch_dim(dummy_action)
  dummy_obs = jax_utils.add_batch_dim(dummy_obs)

  policy = hk.without_apply_rng(hk.transform(_actor_fn, apply_rng=True))
  critic = hk.without_apply_rng(hk.transform(_critic_fn, apply_rng=True))

  return {
      'policy':
          networks_lib.FeedForwardNetwork(
              lambda key: policy.init(key, dummy_obs), policy.apply),
      'critic':
          networks_lib.FeedForwardNetwork(
              lambda key: critic.init(key, dummy_obs, dummy_action),
              critic.apply),
  }
