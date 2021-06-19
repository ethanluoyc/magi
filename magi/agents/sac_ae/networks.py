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
        name=None,
    ):
        super().__init__(name=name)
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


class Policy(hk.Module):
    """
    Policy for SAC.
    """

    def __init__(
        self,
        action_size,
        hidden_sizes=(256, 256),
        log_std_min=-20.0,
        log_std_max=2.0,
        clip_log_std=True,
        name=None,
    ):
        super().__init__(name=name)
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.clip_log_std = clip_log_std

    def __call__(self, x):
        x = MLP(
            2 * self.action_size,
            self.hidden_sizes,
            hidden_activation=nn.relu,
            hidden_scale=jnp.sqrt(2.0),
        )(x)
        mean, log_std = jnp.split(x, 2, axis=1)
        if self.clip_log_std:
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        else:
            # From Dennis Yarats
            # https://github.com/denisyarats/pytorch_sac/blob/929cc6e7efbe7637c2ee1c43e2e7d4b3ad223523/agent/actor.py#L77
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
                jnp.tanh(log_std) + 1.0
            )
        return GaussianTanhTransformedHead(mean, log_std)


class Encoder(hk.Module):
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
        w_init = DeltaOrthogonal(scale=np.sqrt(2.0))
        x = hk.Conv2D(
            self.num_filters, kernel_shape=4, stride=2, padding="VALID", w_init=w_init
        )(x)
        x = self._activation(x)
        for _ in range(self.num_layers - 1):
            x = hk.Conv2D(
                self.num_filters,
                kernel_shape=3,
                stride=1,
                padding="VALID",
                w_init=w_init,
            )(x)
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

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x):
        w_init = hk.initializers.Orthogonal(scale=1.0)
        x = hk.Linear(self.output_size, w_init=w_init)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jnp.tanh(x)
        return x


class Critic(hk.Module):
    """
    Critic for DDPG, TD3 and SAC.
    """

    def __init__(self, hidden_sizes=(256, 256), name=None):
        super().__init__(name)
        self.hidden_sizes = hidden_sizes

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                1,
                self.hidden_sizes,
                hidden_activation=nn.relu,
                hidden_scale=jnp.sqrt(2.0),
            )(x)

        x = jnp.concatenate([s, a], axis=-1)
        # Return list even if num_critics == 1 for simple implementation.
        return [_fn(x).squeeze(-1) for _ in range(2)]


class Decoder(hk.Module):
    """
    Decoder for SAC+AE.
    """

    def __init__(self, output_channels, num_layers=4, num_filters=32):
        super().__init__()
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.num_filters = num_filters
        self._activation = jax.nn.relu
        self.map_size = 43 - 2 * num_layers
        self.last_conv_dim = num_filters * self.map_size * self.map_size

    def __call__(self, x):
        # Apply linear layer.
        w_init = hk.initializers.Orthogonal(scale=jnp.sqrt(2.0))
        x = hk.Linear(self.last_conv_dim, w_init=w_init)(x)
        x = self._activation(x).reshape(
            -1, self.map_size, self.map_size, self.num_filters
        )

        # Apply Transposed CNN.
        w_init = DeltaOrthogonal(scale=np.sqrt(2.0))
        for _ in range(self.num_layers - 1):
            x = hk.Conv2DTranspose(
                self.num_filters,
                kernel_shape=3,
                stride=1,
                padding="VALID",
                w_init=w_init,
            )(x)
            x = self._activation(x)

        # Apply output layer.
        w_init = DeltaOrthogonal(scale=1.0)
        x = hk.Conv2DTranspose(
            self.output_channels,
            kernel_shape=4,
            stride=2,
            padding="VALID",
            w_init=w_init,
        )(x)
        return x


@dataclass
class GaussianTanhTransformedHead:
    mean: jnp.ndarray
    log_std: jnp.ndarray

    def sample(self, seed):
        return reparameterize_gaussian_and_tanh(
            self.mean, self.log_std, seed, return_log_pi=False
        )

    def sample_and_log_prob(self, key):
        return reparameterize_gaussian_and_tanh(
            self.mean, self.log_std, key, return_log_pi=True
        )

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
    Notes:
      There are numerical issues when the action is near the boundaries
      The relu(.) + 1e-6 is used to ensure that the inputs to log() is larger than 1e-6.
      to avoid really small output.
      This is the the trick adapted used in rlljax, pytorch_sac, pytorch_sac_ae.
    """
    return gaussian_log_prob(log_std, noise) - jnp.log(
        nn.relu(1.0 - jnp.square(action)) + 1e-6
    )


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
    critic_hidden_sizes: Sequence[int] = (1024, 1024),
    actor_hidden_sizes: Sequence[int] = (1024, 1024),
    latent_size: int = 50,
    log_std_min: float = -10.0,
    log_std_max: float = 2.0,
    num_filters: int = 32,
    num_layers: int = 4,
):
    def critic(x, a):
        # Define without linear layer.
        # We need to define the linear layer outside the critic because the decoder needs
        # to use the latents from the output AFTER applying the linear layer for
        # reconstruction.
        return Critic(
            hidden_sizes=critic_hidden_sizes,
        )(x, a)

    def actor(x):
        # Define with linear layer.
        # Since the actor path is not needed for reconstruction we can put the linear
        # layer for the actor path here.
        x = SACLinear(output_size=latent_size)(x)
        return Policy(
            action_size=environment_spec.actions.shape[0],
            hidden_sizes=actor_hidden_sizes,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            clip_log_std=False,
        )(x)

    def encoder(x):
        return Encoder(num_filters=num_filters, num_layers=num_layers)(x)

    def linear(x):
        return SACLinear(output_size=latent_size)(x)

    def decoder(x):
        return Decoder(
            environment_spec.observations.shape[-1],
            num_filters=num_filters,
            num_layers=num_layers,
        )(x)

    # Encoder.
    return {
        "encoder": encoder,
        "decoder": decoder,
        "critic": critic,
        "actor": actor,
        "linear": linear,
    }
