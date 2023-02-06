"""Default networks for SAC actor and critic."""
from typing import Sequence

import haiku as hk
import jax.numpy as jnp
import numpy as onp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils


class GaussianPolicy(hk.Module):
    """
    Policy for SAC.
    """

    def __init__(self, action_size, hidden_units=(256, 256), name=None):
        super().__init__(name=name)
        self.hidden_units = hidden_units
        self.action_size = action_size

    def __call__(self, x):
        torso = networks_lib.LayerNormMLP(
            layer_sizes=self.hidden_units, activate_final=True
        )
        h = torso(x)
        return networks_lib.NormalTanhDistribution(self.action_size)(h)


class DoubleCritic(hk.Module):
    """Critic for SAC."""

    def __init__(self, hidden_units=(256, 256), name=None):
        super().__init__(name=name)
        self.num_critics = 2
        self.hidden_units = hidden_units

    def __call__(self, s, a):
        def _fn(x):
            torso = networks_lib.LayerNormMLP(
                layer_sizes=self.hidden_units, activate_final=True
            )
            head = hk.Linear(1)
            return head(torso(x))

        x = jnp.concatenate([s, a], axis=-1)
        # Return list even if num_critics == 1 for simple implementation.
        return [_fn(x).squeeze(-1) for _ in range(self.num_critics)]


def apply_policy_sample(networks, eval_mode: bool):
    def policy_network(params, key, obs):
        action_dist = networks["policy"].apply(params, obs)
        return action_dist.mode() if eval_mode else action_dist.sample(seed=key)

    return policy_network


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256),
    critic_layer_sizes: Sequence[int] = (256, 256),
):
    num_dimensions = onp.prod(spec.actions.shape, dtype=int)

    def _critic_fn(obs, a):
        return DoubleCritic(hidden_units=critic_layer_sizes)(obs, a)

    def _policy_fn(obs):
        return GaussianPolicy(
            action_size=num_dimensions, hidden_units=policy_layer_sizes
        )(obs)

    dummy_action = utils.zeros_like(spec.actions)
    dummy_obs = utils.zeros_like(spec.observations)
    dummy_action = utils.add_batch_dim(dummy_action)
    dummy_obs = utils.add_batch_dim(dummy_obs)

    policy = hk.without_apply_rng(hk.transform(_policy_fn, apply_rng=True))
    critic = hk.without_apply_rng(hk.transform(_critic_fn, apply_rng=True))

    return {
        "policy": networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs), policy.apply
        ),
        "critic": networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply
        ),
    }
