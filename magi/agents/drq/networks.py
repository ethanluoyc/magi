from dataclasses import dataclass
import math
from typing import Sequence

import haiku as hk
import jax
from jax import nn
import jax.numpy as jnp
import numpy as np
import distrax

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
    init_fn = jax.nn.initializers.delta_orthogonal(self.scale, self.axis)
    return init_fn(hk.next_rng_key(), shape, dtype)


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


class StateDependentGaussianPolicy(hk.Module):
  """
    Policy for SAC.
    """

  def __init__(self,
               action_size,
               hidden_units=(256, 256),
               log_std_min=-20.0,
               log_std_max=2.0,
               clip_log_std=True,
               temperature=1.0):
    super().__init__()
    self.action_size = action_size
    self.hidden_units = hidden_units
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    self.clip_log_std = clip_log_std
    self.temperature = temperature

  def __call__(self, x):
    x = MLP(2 * self.action_size,
            self.hidden_units,
            hidden_activation=nn.relu,
            hidden_scale=jnp.sqrt(2.),
            output_scale=jnp.sqrt(2.))(x)
    mean, log_std = jnp.split(x, 2, axis=1)
    log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
    base_dist = distrax.MultivariateNormalDiag(loc=mean,
                                               scale_diag=jnp.exp(log_std)
                                               * self.temperature)
    return distrax.Transformed(distribution=base_dist,
                               bijector=distrax.Block(distrax.Tanh(), 1))


class SACEncoder(hk.Module):
  """
    Encoder for SAC+AE.
    """

  def __init__(self, num_layers=4, num_filters=32):
    super().__init__()
    self.num_layers = num_layers
    self.num_filters = num_filters
    self._activation = jax.nn.relu

  def __call__(self, x):
    # Floatify the image.
    x = x.astype(jnp.float32) / 255.0

    # Apply CNN.
    w_init = hk.initializers.Orthogonal(scale=np.sqrt(2.))
    x = hk.Conv2D(self.num_filters,
                  kernel_shape=3,
                  stride=2,
                  padding="VALID",
                  w_init=w_init)(x)
    x = self._activation(x)
    for _ in range(self.num_layers - 1):
      x = hk.Conv2D(self.num_filters,
                    kernel_shape=3,
                    stride=1,
                    padding="VALID",
                    w_init=w_init)(x)
      x = self._activation(x)
    # Flatten the feature map.
    return hk.Flatten()(x)


class SACLinear(hk.Module):
  """
    Linear layer for SAC+AE.
    This includes a linear layer, followed by layer normalization and tanh.
    The block is used to project the features from the CNN encoder to the latent space
    used by the actor and critic. Note that in SAC_AE, there are two of these blocks
    which are used by the actor and critic respectively and they do not share weights.
  """

  def __init__(self, feature_dim):
    super().__init__()
    self.feature_dim = feature_dim

  def __call__(self, x):
    x = hk.Linear(self.feature_dim)(x)
    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
    x = jnp.tanh(x)
    return x


class ContinuousQFunction(hk.Module):
  """
    Critic for DDPG, TD3 and SAC.
    """

  def __init__(self, num_critics=2, hidden_units=(256, 256), name=None):
    super().__init__(name)
    self.num_critics = num_critics
    self.hidden_units = hidden_units

  def __call__(self, s, a):

    def _fn(x):
      return MLP(
          1,
          self.hidden_units,
          hidden_activation=nn.relu,
          hidden_scale=jnp.sqrt(2.),
      )(x)

    x = jnp.concatenate([s, a], axis=-1)
    # Return list even if num_critics == 1 for simple implementation.
    return [_fn(x).squeeze(-1) for _ in range(self.num_critics)]


def make_default_networks(
    environment_spec,
    num_critics: int = 2,
    critic_hidden_sizes: Sequence[int] = (256, 256),
    actor_hidden_sizes: Sequence[int] = (256, 256),
    latent_size: int = 50,
    log_std_min: float = -10.,
    log_std_max: float = 2.,
    num_filters: int = 32,
    num_layers: int = 4,
):

  def critic(x, a):
    x = SACLinear(feature_dim=latent_size)(x)
    return ContinuousQFunction(
        num_critics=num_critics,
        hidden_units=critic_hidden_sizes,
    )(x, a)

  def actor(x):
    x = SACLinear(feature_dim=latent_size)(x)
    return StateDependentGaussianPolicy(
        action_size=environment_spec.actions.shape[0],
        hidden_units=actor_hidden_sizes,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        clip_log_std=True,
    )(x)

  def encoder(x):
    return SACEncoder(num_filters=num_filters, num_layers=num_layers)(x)

  # Encoder.
  return {
      "encoder": encoder,
      "critic": critic,
      "actor": actor,
  }
