"""Default network architectures for TD3."""
from typing import Dict, Sequence

from acme import specs
from acme.jax import networks
from acme.jax import utils
import haiku as hk
import jax.numpy as jnp
import numpy as np


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256),
    critic_layer_sizes: Sequence[int] = (256, 256),
) -> Dict[str, hk.Transformed]:
    """Make default networks used by TD3."""
    action_size = np.prod(spec.actions.shape, dtype=int)

    def _critic(h):
        output = hk.Sequential(
            [
                hk.nets.MLP(critic_layer_sizes, activate_final=True),
                hk.Linear(1),
            ]
        )(h)
        return jnp.squeeze(output, axis=-1)

    def _double_critic(obs, a):
        h = jnp.concatenate([obs, a], axis=-1)
        q1 = _critic(h)
        q2 = _critic(h)
        return q1, q2

    def _policy(obs):
        return hk.Sequential(
            [
                hk.nets.MLP(policy_layer_sizes, activate_final=True),
                hk.Linear(action_size),
                networks.TanhToSpec(spec.actions),
            ]
        )(obs)

    critic = hk.without_apply_rng(hk.transform(_double_critic))
    policy = hk.without_apply_rng(hk.transform(_policy))
    # Create dummy observations and actions to create network parameters.
    dummy_action = utils.zeros_like(spec.actions)
    dummy_obs = utils.zeros_like(spec.observations)
    dummy_action = utils.add_batch_dim(dummy_action)
    dummy_obs = utils.add_batch_dim(dummy_obs)

    return {
        "policy": networks.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs), policy.apply
        ),
        "critic": networks.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply
        ),
    }
