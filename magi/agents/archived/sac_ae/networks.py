from typing import Sequence

from acme.jax.networks import distributional
import haiku as hk
import jax
from jax import nn
import jax.numpy as jnp
import tensorflow_probability

hk_init = hk.initializers
tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions

# uniform_initializer = hk.initializers.VarianceScaling(
#     distribution='uniform', mode='fan_out', scale=0.333)
uniform_initializer = hk_init.UniformScaling(scale=0.333)
glorot_uniform_initializer = hk_init.VarianceScaling(1.0, "fan_avg", "uniform")

# class MLP(hk.Module):

#   def __init__(
#       self,
#       output_dim,
#       hidden_units,
#       hidden_activation=nn.relu,
#       output_activation=None,
#       hidden_scale=1.0,
#       output_scale=1.0,
#   ):
#     super(MLP, self).__init__()
#     self.output_dim = output_dim
#     self.hidden_units = hidden_units
#     self.hidden_activation = hidden_activation
#     self.output_activation = output_activation
#     self.hidden_kwargs = {"w_init": hk.initializers.Orthogonal(scale=hidden_scale)}
#     self.output_kwargs = {"w_init": hk.initializers.Orthogonal(scale=output_scale)}

#   def __call__(self, x):
#     # x_input = x
#     for i, unit in enumerate(self.hidden_units):
#       x = hk.Linear(unit, **self.hidden_kwargs)(x)
#       x = self.hidden_activation(x)
#     x = hk.Linear(self.output_dim, **self.output_kwargs)(x)
#     if self.output_activation is not None:
#       x = self.output_activation(x)
#     return x


class Encoder(hk.Module):

  def __call__(self, observation):
    observation = observation.astype(jnp.float32) / 255.0
    torso = hk.Sequential([
        hk.Conv2D(16, kernel_shape=3, stride=2),
        jax.nn.relu,
        hk.Conv2D(16, kernel_shape=3, stride=2),
        jax.nn.relu,
    ])
    feature = torso(observation)
    return hk.Flatten()(feature)


class Decoder(hk.Module):

  def __call__(self, feature):
    return hk.Sequential([
        hk.Linear(16 * 16 * 16),
        lambda x: jnp.reshape(x, (-1, 16, 16, 16)),
        hk.Conv2DTranspose(16, kernel_shape=3, stride=2),
        jax.nn.relu,
        hk.Conv2DTranspose(3, kernel_shape=3, stride=2),
        jax.nn.relu,
    ])(feature)


class Policy(hk.Module):

  def __init__(self, action_dim: int, name=None):
    super().__init__(name=name)
    self._action_dim = action_dim
    self.log_std_min = -20
    self.log_std_max = 2

  def __call__(self, x):
    x = hk.nets.MLP([512, 512],
                    w_init=uniform_initializer,
                    activation=nn.elu,
                    activate_final=True)(x)
    return distributional.NormalTanhDistribution(self._action_dim)(x)
    # mean, log_std = jnp.split(x, 2, axis=-1)
    # log_std = self.log_std_min + 0.5 * (self.log_std_max -
    #                                     self.log_std_min) * (jnp.tanh(log_std) + 1.0)

    # distribution = tfd.Normal(mean, jnp.exp(log_std))
    # return tfd.Independent(distributional.TanhTransformedDistribution(distribution),
    #                        reinterpreted_batch_ndims=1)


class Critic(hk.Module):

  def __call__(self, feature, action):

    def fn(x):
      torso = hk.nets.MLP([512, 512],
                          w_init=uniform_initializer,
                          activation=nn.elu,
                          activate_final=True)
      linear = hk.Linear(1)
      return linear(torso(x)).squeeze(-1)

    x = jnp.concatenate([feature, action], axis=-1)
    return fn(x), fn(x)


class DeltaOrthogonal(hk.initializers.Initializer):
  """
    Delta-orthogonal initializer.
    """

  def __init__(self, scale=1.0, axis=-1):
    self.scale = scale
    self.axis = axis

  def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Delta orthogonal initializer requires 3D, 4D or 5D shape.")
    w_mat = jnp.zeros(shape, dtype=dtype)
    w_orthogonal = hk.initializers.Orthogonal(self.scale, self.axis)(shape[-2:], dtype)
    if len(shape) == 3:
      k = shape[0]
      return jax.ops.index_update(
          w_mat,
          jax.ops.index[(k - 1) // 2, ...],
          w_orthogonal,
      )
    elif len(shape) == 4:
      k1, k2 = shape[:2]
      return jax.ops.index_update(
          w_mat,
          jax.ops.index[(k1 - 1) // 2, (k2 - 1) // 2, ...],
          w_orthogonal,
      )
    else:
      k1, k2, k3 = shape[:3]
      return jax.ops.index_update(
          w_mat,
          jax.ops.index[(k1 - 1) // 2, (k2 - 1) // 2, (k3 - 1) // 2, ...],
          w_orthogonal,
      )


class SACEncoder(hk.Module):
  """
    Encoder for SAC+AE.
    """

  def __init__(self, num_layers=2, feature_dim=50, num_filters=32, negative_slope=0.1):
    super().__init__()
    self.num_layers = num_layers
    self.num_filters = num_filters
    self.negative_slope = negative_slope
    self.feature_dim = feature_dim

  def __call__(self, x):
    # Floatify the image.
    x = x.astype(jnp.float32) / 255.0

    # Apply CNN.
    # w_init = DeltaOrthogonal(scale=np.sqrt(2 / (1 + self.negative_slope**2)))
    x = hk.Conv2D(self.num_filters,
                  kernel_shape=4,
                  stride=2,
                  padding="VALID",
                  w_init=glorot_uniform_initializer)(x)
    x = nn.elu(x)
    for _ in range(self.num_layers - 1):
      x = hk.Conv2D(self.num_filters,
                    kernel_shape=3,
                    stride=1,
                    padding="VALID",
                    w_init=glorot_uniform_initializer)(x)
      x = nn.elu(x)
    return x


class SACLinear(hk.Module):

  def __init__(self, feature_dim, name=None):
    super().__init__(name=name)
    self.feature_dim = feature_dim

  def __call__(self, x):
    # w_init = hk.initializers.Orthogonal(scale=1.0)
    x = hk.Flatten()(x)
    fc = hk.Linear(self.feature_dim, w_init=uniform_initializer)
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return jnp.tanh(ln(fc(x)))


class SACDecoder(hk.Module):
  """
    Decoder for SAC+AE.
    """

  def __init__(self, num_channels, num_layers=2, num_filters=32, negative_slope=0.1):
    super().__init__()
    self.num_layers = num_layers
    self.num_filters = num_filters
    self.negative_slope = negative_slope
    self.map_size = 43 - 2 * num_layers
    self.last_conv_dim = num_filters * self.map_size * self.map_size
    self._num_channels = num_channels

  def __call__(self, x):
    # Apply linear layer.
    # w_init = hk.initializers.Orthogonal(scale=np.sqrt(2 /
    # (1 + self.negative_slope**2)))
    x = hk.Linear(self.last_conv_dim, w_init=uniform_initializer)(x)
    x = nn.elu(x).reshape(-1, self.map_size, self.map_size, self.num_filters)

    # Apply Transposed CNN.
    # w_init = DeltaOrthogonal(scale=np.sqrt(2 / (1 + self.negative_slope**2)))
    for _ in range(self.num_layers - 1):
      x = hk.Conv2DTranspose(self.num_filters,
                             kernel_shape=3,
                             stride=1,
                             padding="VALID",
                             w_init=glorot_uniform_initializer)(x)
      x = nn.elu(x)

    # Apply output layer.
    # w_init = DeltaOrthogonal(scale=1.0)
    x = hk.Conv2DTranspose(self._num_channels,
                           kernel_shape=4,
                           stride=2,
                           padding="VALID",
                           w_init=glorot_uniform_initializer)(x)
    return x
