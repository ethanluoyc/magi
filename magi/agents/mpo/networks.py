"""Default networks for MPO agent"""
from typing import Sequence

import haiku as hk
import jax.numpy as jnp
import numpy as onp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils


def apply_policy_and_sample(networks, action_spec, eval_mode: bool):
    def policy_network(params, key, obs):
        action_dist = networks["policy"].apply(params, obs)
        action = action_dist.mode() if eval_mode else action_dist.sample(seed=key)
        return jnp.clip(action, action_spec.minimum, action_spec.maximum)

    return policy_network


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
):
    num_dimensions = onp.prod(spec.actions.shape, dtype=int)

    def _critic_fn(obs, a):
        network = networks_lib.CriticMultiplexer(
            critic_network=networks_lib.LayerNormMLP(list(critic_layer_sizes) + [1]),
            # The policy network used in MPO by default may be out of bounds.
            # Here, we clip the actions following what's been done
            # in the original MPO implementation.
            action_network=networks_lib.ClipToSpec(spec.actions),
        )
        q = network(obs, a)
        return jnp.squeeze(q, axis=-1)

    def _policy_fn(obs):
        return hk.Sequential(
            [
                networks_lib.LayerNormMLP(
                    list(policy_layer_sizes), activate_final=True
                ),
                networks_lib.MultivariateNormalDiagHead(num_dimensions),
            ]
        )(obs)

    dummy_action = utils.zeros_like(spec.actions)
    dummy_obs = utils.zeros_like(spec.observations)
    dummy_action = utils.add_batch_dim(dummy_action)
    dummy_obs = utils.add_batch_dim(dummy_obs)

    policy = hk.without_apply_rng(hk.transform(_policy_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))

    return {
        "policy": networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs), policy.apply
        ),
        "critic": networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply
        ),
    }
