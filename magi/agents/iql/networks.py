"""Default networks used by the IQL agent."""
import dataclasses
from typing import Callable, Optional, Sequence, Tuple

from acme import specs
import haiku as hk
from jax import nn
import jax.numpy as jnp
import numpy as onp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


# The default uses onp as we do not want to trigger JAX init
def _default_init(scale: Optional[float] = onp.sqrt(2)):
    return hk.initializers.Orthogonal(scale)


class MLP(hk.Module):
    """MLP with dropout"""

    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        activate_final: int = False,
        dropout_rate: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = hk.Linear(size, w_init=_default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
        return x


class NormalTanhPolicy(hk.Module):
    """Gaussian policy."""

    def __init__(
        self,
        hidden_dims: Sequence[int],
        action_dim: int,
        state_dependent_std: bool = True,
        log_std_scale: float = 1.0,
        log_std_min: Optional[float] = None,
        log_std_max: Optional[float] = None,
        tanh_squash_distribution: bool = True,
        name=None,
    ):
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        self.log_std_scale = log_std_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash_distribution = tanh_squash_distribution

    def __call__(self, observations: jnp.ndarray) -> tfd.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = hk.Linear(self.action_dim, w_init=_default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, w_init=_default_init(self.log_std_scale)
            )(outputs)
        else:
            log_stds = hk.get_parameter("log_stds", (self.action_dim,), init=jnp.zeros)

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(
                distribution=base_dist, bijector=tfb.Tanh()
            )
        else:
            return base_dist


class ValueCritic(hk.Module):
    """Value network."""

    def __init__(self, hidden_dims: Sequence[int]):
        super().__init__()
        self.hidden_dims = hidden_dims

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(hk.Module):
    """A single critic network."""

    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activations = activations

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(hk.Module):
    """Double critic network."""

    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activations = activations

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


def apply_policy_and_sample(
    networks: IQLNetworks, action_spec: specs.BoundedArray, eval_mode: bool = False
):
    def policy_network(params, key, obs):
        action_dist = networks.policy_network.apply(params, obs)
        action = action_dist.mode() if eval_mode else action_dist.sample(seed=key)
        return jnp.clip(action, action_spec.minimum, action_spec.maximum)

    return policy_network


def make_networks(
    spec: specs.EnvironmentSpec,
    hidden_dims=(256, 256),
):
    action_dim = spec.actions.shape[-1]

    def _actor_fn(observations):
        return NormalTanhPolicy(
            hidden_dims,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            state_dependent_std=False,
            tanh_squash_distribution=False,
        )(observations)

    def _critic_fn(o, a):
        return DoubleCritic(hidden_dims)(o, a)

    def _value_fn(o):
        return ValueCritic(hidden_dims)(o)

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))
    value = hk.without_apply_rng(hk.transform(_value_fn))
    dummy_obs = jnp.zeros((1, *spec.observations.shape), dtype=spec.observations.dtype)
    dummy_action = jnp.zeros((1, *spec.actions.shape), dtype=spec.actions.dtype)

    return IQLNetworks(
        policy_network=FeedforwardNetwork(
            init=lambda key: policy.init(key, dummy_obs), apply=policy.apply
        ),
        critic_network=FeedforwardNetwork(
            init=lambda key: critic.init(key, dummy_obs, dummy_action),
            apply=critic.apply,
        ),
        value_network=FeedforwardNetwork(
            init=lambda key: value.init(key, dummy_obs),
            apply=value.apply,
        ),
    )
