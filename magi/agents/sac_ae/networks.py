from dataclasses import dataclass
import math
from typing import Sequence

import haiku as hk
import jax
from jax import nn
import jax.numpy as jnp
import numpy as np


class MLP(hk.Module):

  def __init__(
      self,
      output_dim,
      hidden_units,
      hidden_activation=nn.relu,
      output_activation=None,
      hidden_scale=1.0,
      output_scale=1.0,
  ):
    super().__init__()
    self.output_dim = output_dim
    self.hidden_units = hidden_units
    self.hidden_activation = hidden_activation
    self.output_activation = output_activation
    self.hidden_kwargs = {"w_init": hk.initializers.Orthogonal(scale=hidden_scale)}
    self.output_kwargs = {"w_init": hk.initializers.Orthogonal(scale=output_scale)}

  def __call__(self, x):
    for unit in self.hidden_units:
      x = hk.Linear(unit, **self.hidden_kwargs)(x)
      x = self.hidden_activation(x)
    x = hk.Linear(self.output_dim, **self.output_kwargs)(x)
    if self.output_activation is not None:
      x = self.output_activation(x)
    return x


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


class StateDependentGaussianPolicy(hk.Module):
  """
    Policy for SAC.
    """

  def __init__(
      self,
      action_size,
      hidden_units=(256, 256),
      log_std_min=-20.0,
      log_std_max=2.0,
      clip_log_std=True,
  ):
    super().__init__()
    self.action_size = action_size
    self.hidden_units = hidden_units
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    self.clip_log_std = clip_log_std

  def __call__(self, x):
    x = MLP(
        2 * self.action_size,
        self.hidden_units,
        hidden_activation=nn.relu,
    )(x)
    mean, log_std = jnp.split(x, 2, axis=1)
    if self.clip_log_std:
      log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
    else:
      log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
          jnp.tanh(log_std) + 1.0)
    return GaussianTanhTransformedHead(mean, log_std)


class SACEncoder(hk.Module):
  """
    Encoder for SAC+AE.
    """

  def __init__(self, num_layers=4, num_filters=32, negative_slope=0.1):
    super().__init__()
    self.num_layers = num_layers
    self.num_filters = num_filters
    self.negative_slope = negative_slope

  def __call__(self, x):
    # Floatify the image.
    x = x.astype(jnp.float32) / 255.0

    # Apply CNN.
    w_init = DeltaOrthogonal(scale=np.sqrt(2 / (1 + self.negative_slope**2)))
    x = hk.Conv2D(self.num_filters,
                  kernel_shape=4,
                  stride=2,
                  padding="VALID",
                  w_init=w_init)(x)
    x = nn.leaky_relu(x, self.negative_slope)
    for _ in range(self.num_layers - 1):
      x = hk.Conv2D(self.num_filters,
                    kernel_shape=3,
                    stride=1,
                    padding="VALID",
                    w_init=w_init)(x)
      x = nn.leaky_relu(x, self.negative_slope)
    # Flatten the feature map.
    return hk.Flatten()(x)


class SACLinear(hk.Module):
  """
    Linear layer for SAC+AE.
    """

  def __init__(self, feature_dim):
    super().__init__()
    self.feature_dim = feature_dim

  def __call__(self, x):
    w_init = hk.initializers.Orthogonal(scale=1.0)
    x = hk.Linear(self.feature_dim, w_init=w_init)(x)
    x = hk.LayerNorm(axis=1, create_scale=True, create_offset=True)(x)
    x = jnp.tanh(x)
    return x


class ContinuousQFunction(hk.Module):
  """
    Critic for DDPG, TD3 and SAC.
    """

  def __init__(
      self,
      num_critics=2,
      hidden_units=(256, 256),
      name=None
  ):
    super().__init__(name)
    self.num_critics = num_critics
    self.hidden_units = hidden_units

  def __call__(self, s, a):

    def _fn(x):
      return MLP(
          1,
          self.hidden_units,
          hidden_activation=nn.relu,
          hidden_scale=np.sqrt(2),
      )(x)

    x = jnp.concatenate([s, a], axis=1)
    # Return list even if num_critics == 1 for simple implementation.
    return [_fn(x).squeeze(-1) for _ in range(self.num_critics)]


class SACDecoder(hk.Module):
  """
    Decoder for SAC+AE.
    """

  def __init__(self, state_shape, num_layers=4, num_filters=32, negative_slope=0.1):
    super().__init__()
    self.state_shape = state_shape
    self.num_layers = num_layers
    self.num_filters = num_filters
    self.negative_slope = negative_slope
    self.map_size = 43 - 2 * num_layers
    self.last_conv_dim = num_filters * self.map_size * self.map_size

  def __call__(self, x):
    # Apply linear layer.
    w_init = hk.initializers.Orthogonal(scale=np.sqrt(2 / (1 + self.negative_slope**2)))
    x = hk.Linear(self.last_conv_dim, w_init=w_init)(x)
    x = nn.leaky_relu(x, self.negative_slope).reshape(-1, self.map_size, self.map_size,
                                                      self.num_filters)

    # Apply Transposed CNN.
    w_init = DeltaOrthogonal(scale=np.sqrt(2 / (1 + self.negative_slope**2)))
    for _ in range(self.num_layers - 1):
      x = hk.Conv2DTranspose(self.num_filters,
                             kernel_shape=3,
                             stride=1,
                             padding="VALID",
                             w_init=w_init)(x)
      x = nn.leaky_relu(x, self.negative_slope)

    # Apply output layer.
    w_init = DeltaOrthogonal(scale=1.0)
    x = hk.Conv2DTranspose(self.state_shape.shape[2],
                           kernel_shape=4,
                           stride=2,
                           padding="VALID",
                           w_init=w_init)(x)
    return x


class SLACEncoder(hk.Module):
  """
    Encoder for SLAC.
    """

  def __init__(self, output_dim=256, negative_slope=0.2):
    super().__init__()
    self.output_dim = output_dim
    self.negative_slope = negative_slope

  def __call__(self, x):
    B, S, H, W, C = x.shape  # pylint: disable=invalid-name

    # Floatify the image.
    x = x.astype(jnp.float32) / 255.0
    # Reshape.
    x = x.reshape([B * S, H, W, C])
    # Apply CNN.
    w_init = DeltaOrthogonal(scale=1.0)
    depth = [32, 64, 128, 256, self.output_dim]
    kernel = [5, 3, 3, 3, 4]
    stride = [2, 2, 2, 2, 1]
    padding = ["SAME", "SAME", "SAME", "SAME", "VALID"]

    for i in range(5):
      x = hk.Conv2D(
          depth[i],
          kernel_shape=kernel[i],
          stride=stride[i],
          padding=padding[i],
          w_init=w_init,
      )(x)
      x = nn.leaky_relu(x, self.negative_slope)

    return x.reshape([B, S, -1])


class SLACDecoder(hk.Module):
  """
    Decoder for SLAC.
    """

  def __init__(self, state_space, std=1.0, negative_slope=0.2):
    super().__init__()
    self.state_space = state_space
    self.std = std
    self.negative_slope = negative_slope

  def __call__(self, x):
    B, S, latent_dim = x.shape  # pylint: disable=invalid-name

    # Reshape.
    x = x.reshape([B * S, 1, 1, latent_dim])

    # Apply CNN.
    w_init = DeltaOrthogonal(scale=1.0)
    depth = [256, 128, 64, 32, self.state_space.shape[2]]
    kernel = [4, 3, 3, 3, 5]
    stride = [1, 2, 2, 2, 2]
    padding = ["VALID", "SAME", "SAME", "SAME", "SAME"]

    for i in range(4):
      x = hk.Conv2DTranspose(
          depth[i],
          kernel_shape=kernel[i],
          stride=stride[i],
          padding=padding[i],
          w_init=w_init,
      )(x)
      x = nn.leaky_relu(x, self.negative_slope)

    x = hk.Conv2DTranspose(
        depth[-1],
        kernel_shape=kernel[-1],
        stride=stride[-1],
        padding=padding[-1],
        w_init=w_init,
    )(x)

    _, W, H, C = x.shape  # pylint: disable=invalid-name
    x = x.reshape([B, S, W, H, C])
    return x, jax.lax.stop_gradient(jnp.ones_like(x) * self.std)


@dataclass
class GaussianTanhTransformedHead:
  mean: jnp.ndarray
  log_std: jnp.ndarray

  def sample(self, seed):
    return reparameterize_gaussian_and_tanh(self.mean,
                                            self.log_std,
                                            seed,
                                            return_log_pi=False)

  def sample_and_log_prob(self, key):
    return reparameterize_gaussian_and_tanh(self.mean,
                                            self.log_std,
                                            key,
                                            return_log_pi=True)

  def mode(self):
    return jnp.tanh(self.mean)


@jax.jit
def gaussian_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
  """
    Calculate log probabilities of gaussian distributions.
    """
  return -0.5 * (jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi))


@jax.jit
def gaussian_and_tanh_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
  """
    Calculate log probabilities of gaussian distributions and tanh transformation.
    """
  return gaussian_log_prob(log_std,
                           noise) - jnp.log(nn.relu(1.0 - jnp.square(action)) + 1e-6)


def reparameterize_gaussian_and_tanh(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jnp.ndarray,
    return_log_pi: bool = True,
):
  """
    Sample from gaussian distributions and tanh transforamation.
    """
  std = jnp.exp(log_std)
  noise = jax.random.normal(key, std.shape)
  action = jnp.tanh(mean + noise * std)
  if return_log_pi:
    return action, gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=-1)
  else:
    return action


def make_default_networks(
    environment_spec,
    num_critics: int = 2,
    critic_hidden_sizes: Sequence[int] = (1024, 1024),
    actor_hidden_sizes: Sequence[int] = (1024, 1024),
    latent_size: int = 50,
    log_std_min: float = -10.,
    log_std_max: float = 2.,
    num_filters: int = 32,
    num_layers: int = 4,
):

  def critic(x, a):
    # Define without linear layer.
    return ContinuousQFunction(
        num_critics=num_critics,
        hidden_units=critic_hidden_sizes,
    )(x, a)

  def actor(x):
    # Define with linear layer.
    x = SACLinear(feature_dim=latent_size)(x)
    return StateDependentGaussianPolicy(
        action_size=environment_spec.actions.shape[0],
        hidden_units=actor_hidden_sizes,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        clip_log_std=False,
    )(x)

  def encoder(x):
    return SACEncoder(num_filters=num_filters, num_layers=num_layers)(x)

  def linear(x):
    return SACLinear(feature_dim=latent_size)(x)

  def decoder(x):
    return SACDecoder(environment_spec.observations,
                      num_filters=num_filters,
                      num_layers=num_layers)(x)

  # Encoder.
  return {
      "encoder": encoder,
      "decoder": decoder,
      "critic": critic,
      "actor": actor,
      "linear": linear
  }
