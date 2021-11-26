"""Default networks used by the IQL agent."""
import dataclasses
from typing import Callable, Optional, Sequence, Tuple

from acme import specs
from acme.jax import networks as networks_lib
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


class Policy(hk.Module):
    """Policy network for IQL."""

    def __init__(
        self,
        hidden_dims: Sequence[int],
        action_dim: int,
        state_dependent_std: bool = True,
        log_std_scale: float = 1.0,
        log_std_min: Optional[float] = None,
        log_std_max: Optional[float] = None,
        tanh_squash_distribution: bool = True,
        dropout_rate: Optional[float] = None,
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
        self.dropout_rate = dropout_rate

    def __call__(
        self,
        observations: jnp.ndarray,
        is_training: bool,
        rng: Optional[jnp.ndarray] = None,
    ) -> tfd.Distribution:
        torso = hk.nets.MLP(
            output_sizes=self.hidden_dims,
            activate_final=True,
            w_init=_default_init(),
        )
        if is_training and self.dropout_rate is not None:
            outputs = torso(observations, dropout_rate=self.dropout_rate, rng=rng)
        else:
            outputs = torso(observations)

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
        critic_value = hk.nets.MLP(
            output_sizes=(*self.hidden_dims, 1),
            w_init=_default_init(),
        )(observations)
        return jnp.squeeze(critic_value, -1)


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
        critic = hk.nets.MLP(
            output_sizes=(*self.hidden_dims, 1),
            w_init=_default_init(),
        )(inputs)
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
class IQLNetworks:
    policy_network: networks_lib.FeedForwardNetwork
    critic_network: networks_lib.FeedForwardNetwork
    value_network: networks_lib.FeedForwardNetwork


def apply_policy_and_sample(
    networks: IQLNetworks,
    action_spec: specs.BoundedArray,
    eval_mode: bool = False,
):
    def policy_network(params, key, obs):
        # Note: the policy network is only used for inference, regardless
        # of whether we are sampling actions or using the mode. Thus
        # is_training is False
        action_dist = networks.policy_network.apply(params, obs, is_training=False)
        action = action_dist.mode() if eval_mode else action_dist.sample(seed=key)
        return jnp.clip(action, action_spec.minimum, action_spec.maximum)

    return policy_network


def make_networks(
    spec: specs.EnvironmentSpec,
    hidden_dims=(256, 256),
    dropout_rate: Optional[float] = None,
):
    action_dim = spec.actions.shape[-1]

    def _actor_fn(observations, is_training: bool, rng=None):
        # To handle potential use of dropout, the policy used by IQL
        # requires two additional args: `is_training` and `rng`
        # which are used by the IQL learner to optionally enable dropout
        # in learning the policy.
        return Policy(
            hidden_dims,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            state_dependent_std=False,
            tanh_squash_distribution=False,
            dropout_rate=dropout_rate,
        )(observations, is_training, rng=rng)

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
        policy_network=networks_lib.FeedForwardNetwork(
            init=lambda key: policy.init(key, dummy_obs, is_training=False),
            apply=policy.apply,
        ),
        critic_network=networks_lib.FeedForwardNetwork(
            init=lambda key: critic.init(key, dummy_obs, dummy_action),
            apply=critic.apply,
        ),
        value_network=networks_lib.FeedForwardNetwork(
            init=lambda key: value.init(key, dummy_obs),
            apply=value.apply,
        ),
    )
