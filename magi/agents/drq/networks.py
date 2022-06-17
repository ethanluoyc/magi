"""Default networks used by DrQ agent."""
from dataclasses import dataclass
from typing import Optional, Sequence

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# Use onp since we don't want to initialize JAX on imports
orthogonal_init = hk.initializers.Orthogonal(scale=onp.sqrt(2.0))


@dataclass
class GaussianTanhTransformedDistribution:
  """Parametric tanh-squahsed MVN distribution."""

  # TODO(yl): test the head module from acme.
  mean: jnp.ndarray
  log_std: jnp.ndarray

  def _build_dist(self):
    base_dist = tfd.Normal(loc=self.mean, scale=jnp.exp(self.log_std))
    return tfd.Independent(
        tfd.TransformedDistribution(
            distribution=base_dist, bijector=tfb.Tanh()),
        reinterpreted_batch_ndims=1,
    )

  def sample(self, seed):
    return self._build_dist().sample(seed=seed)

  def sample_and_log_prob(self, seed):
    dist = self._build_dist()
    sample = dist.sample(seed=seed)
    return sample, dist.log_prob(sample)

  def mode(self):
    return jnp.tanh(self.mean)


class Encoder(hk.Module):
  """Encoder network used by DrQ-v1"""

  def __init__(self, num_layers=4, num_filters=32, activation=jax.nn.relu):
    super().__init__()
    self.num_layers = num_layers
    self.num_filters = num_filters
    self._activation = activation

  def __call__(self, x):
    # Floatify the image.
    x = x.astype(jnp.float32) / 255.0

    # Apply CNN.
    x = hk.Conv2D(
        self.num_filters,
        kernel_shape=3,
        stride=2,
        padding='VALID',
        w_init=orthogonal_init,
    )(
        x)
    x = self._activation(x)
    for _ in range(self.num_layers - 1):
      x = hk.Conv2D(
          self.num_filters,
          kernel_shape=3,
          stride=1,
          padding='VALID',
          w_init=orthogonal_init,
      )(
          x)
      x = self._activation(x)
    # Flatten the feature map.
    return hk.Flatten()(x)


class Actor(hk.Module):
  """Policy network used by DrQ-v1"""

  def __init__(
      self,
      action_size: int,
      latent_size: int = 50,
      hidden_sizes: Sequence[int] = (256, 256),
      log_std_min: float = -10.0,
      log_std_max: float = 2.0,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.latent_size = latent_size
    self.action_size = action_size
    self.hidden_sizes = hidden_sizes
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max

  def __call__(self, inputs):
    h = hk.Linear(self.latent_size)(inputs)
    # Default epsilon different from flax
    h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
    h = jnp.tanh(h)
    h = hk.nets.MLP(
        output_sizes=self.hidden_sizes,
        w_init=orthogonal_init,
        activate_final=True)(
            h)
    mean_linear = hk.Linear(self.action_size, w_init=orthogonal_init)
    log_std_linear = hk.Linear(self.action_size, w_init=orthogonal_init)
    mean = mean_linear(h)
    log_std = log_std_linear(h)
    log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
    return GaussianTanhTransformedDistribution(mean, log_std)


class Critic(hk.Module):
  """Critic network used by DrQ-v1 agent."""

  def __init__(self, hidden_sizes=(256, 256), name=None):
    super().__init__(name)
    self.hidden_sizes = hidden_sizes

  def __call__(self, observation, action):
    inputs = jnp.concatenate([observation, action], axis=-1)
    q_value = hk.nets.MLP(
        output_sizes=(*self.hidden_sizes, 1),
        w_init=orthogonal_init,
        activate_final=False,
    )(inputs).squeeze(-1)
    return q_value


class DoubleCritic(hk.Module):
  """Twin critic network used by DrQ-v1 agent."""

  def __init__(self, latent_size: int = 50, hidden_sizes=(256, 256), name=None):
    super().__init__(name)
    self.hidden_sizes = hidden_sizes
    self.latent_size = latent_size

  def __call__(self, observation, action):
    h = hk.Linear(self.latent_size)(observation)
    # Default epsilon different from flax
    h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
    h = jnp.tanh(h)
    critic1 = Critic(self.hidden_sizes, name='critic1')
    critic2 = Critic(self.hidden_sizes, name='critic2')
    return critic1(h, action), critic2(h, action)


def apply_policy_sample(networks, eval_mode: bool):
  """Return a pure function for the DrQ-v1 policy"""

  def policy_network(params, key, observation):
    feature_map = networks['encoder'].apply(params['encoder'], observation)
    action_dist = networks['actor'].apply(params['actor'], feature_map)
    return action_dist.mode() if eval_mode else action_dist.sample(seed=key)

  return policy_network


def make_networks(
    spec: specs.EnvironmentSpec,
    critic_hidden_sizes: Sequence[int] = (256, 256),
    actor_hidden_sizes: Sequence[int] = (256, 256),
    latent_size: int = 50,
    log_std_min: float = -10.0,
    log_std_max: float = 2.0,
    num_filters: int = 32,
    num_layers: int = 4,
):
  """Create default neural networks used by DrQ-v1."""

  def _critic(x, a):
    return DoubleCritic(
        latent_size=latent_size,
        hidden_sizes=critic_hidden_sizes,
    )(x, a)

  def _policy(x):
    return Actor(
        action_size=spec.actions.shape[0],
        latent_size=latent_size,
        hidden_sizes=actor_hidden_sizes,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
    )(
        x)

  def _encoder(x):
    return Encoder(num_filters=num_filters, num_layers=num_layers)(x)

  policy = hk.without_apply_rng(hk.transform(_policy, apply_rng=True))
  critic = hk.without_apply_rng(hk.transform(_critic, apply_rng=True))
  encoder = hk.without_apply_rng(hk.transform(_encoder, apply_rng=True))

  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)
  dummy_encoded = hk.testing.transform_and_run(
      _encoder, seed=0, jax_transform=jax.jit)(
          dummy_obs)
  dummy_encoded = jnp.zeros(dummy_encoded.shape, dummy_encoded.dtype)
  # Encoder.
  return {
      'encoder':
          networks_lib.FeedForwardNetwork(
              lambda key: encoder.init(key, dummy_obs), encoder.apply),
      'actor':
          networks_lib.FeedForwardNetwork(
              lambda key: policy.init(key, dummy_encoded), policy.apply),
      'critic':
          networks_lib.FeedForwardNetwork(
              lambda key: critic.init(key, dummy_encoded, dummy_action),
              critic.apply),
  }
