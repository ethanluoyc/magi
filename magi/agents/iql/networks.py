"""Default networks used by the IQL agent."""
import dataclasses
from typing import Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    """MLP with dropout"""

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
        return x


class NormalTanhPolicy(nn.Module):
    """Gaussian policy."""

    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
    ) -> tfd.Distribution:
        outputs = MLP(
            self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate
        )(observations, training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.log_std_scale)
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(
                distribution=base_dist, bijector=tfb.Tanh()
            )
        else:
            return base_dist


class ValueCritic(nn.Module):
    """Value network."""

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    """A single critic network."""

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    """Double critic network."""

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims, activations=self.activations)(
            observations, actions
        )
        critic2 = Critic(self.hidden_dims, activations=self.activations)(
            observations, actions
        )
        return critic1, critic2


@dataclasses.dataclass
class FeedforwardNetwork:
    init: Callable
    apply: Callable


@dataclasses.dataclass
class IQLNetworks:
    policy_network: FeedforwardNetwork
    critic_network: FeedforwardNetwork
    value_network: FeedforwardNetwork


def get_behavior_policy_network(networks: IQLNetworks, temperature: float):
    def policy(params, key, observations):
        dist = networks.policy_network.apply(params, observations, temperature)
        return dist.sample(seed=key)

    return policy


def make_networks(
    observations,
    actions,
    hidden_dims=(256, 256),
    dropout_rate: Optional[float] = None,
):
    action_dim = actions.shape[-1]
    policy_def = NormalTanhPolicy(
        hidden_dims,
        action_dim,
        log_std_scale=1e-3,
        log_std_min=-5.0,
        dropout_rate=dropout_rate,
        state_dependent_std=False,
        tanh_squash_distribution=False,
    )

    critic_def = DoubleCritic(hidden_dims)
    value_def = ValueCritic(hidden_dims)
    dummy_obs = jnp.zeros((1, *observations.shape), dtype=observations.dtype)
    dummy_action = jnp.zeros((1, *actions.shape), dtype=actions.dtype)

    return IQLNetworks(
        policy_network=FeedforwardNetwork(
            init=lambda key: policy_def.init(key, dummy_obs)["params"],
            apply=lambda params, *args, **kwargs: policy_def.apply(
                {"params": params}, *args, **kwargs
            ),
        ),
        critic_network=FeedforwardNetwork(
            init=lambda key: critic_def.init(key, dummy_obs, dummy_action)["params"],
            apply=lambda params, *args, **kwargs: critic_def.apply(
                {"params": params}, *args, **kwargs
            ),
        ),
        value_network=FeedforwardNetwork(
            init=lambda key: value_def.init(key, dummy_obs)["params"],
            apply=lambda params, *args, **kwargs: value_def.apply(
                {"params": params}, *args, **kwargs
            ),
        ),
    )
