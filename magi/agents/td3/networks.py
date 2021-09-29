"""Default network architectures for TD3."""
from typing import Dict, Sequence

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def apply_policy_sample(networks, eval_mode: bool):
    def policy_network(params, key, observation):
        action_mean = networks["policy"].apply(params, observation)
        if eval_mode:
            return action_mean
        else:
            return networks["sample"](action_mean, key)

    return policy_network


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256),
    critic_layer_sizes: Sequence[int] = (256, 256),
    sigma: float = 0.1,
) -> Dict[str, hk.Transformed]:
    """Make default networks used by TD3."""
    action_size = np.prod(spec.actions.shape, dtype=int)

    def _critic(h):
        output = hk.nets.MLP(
            list(critic_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
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
                hk.nets.MLP(
                    policy_layer_sizes,
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activate_final=True,
                ),
                hk.Linear(
                    action_size,
                    hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                ),
                jnp.tanh,
            ]
        )(obs)

    def _sample_fn(action_mean, key):
        exploration_noise = jax.random.normal(key, action_mean.shape) * sigma
        sampled_action = action_mean + exploration_noise
        sampled_action = jnp.clip(
            sampled_action, spec.actions.minimum, spec.actions.maximum
        )
        return sampled_action

    critic = hk.without_apply_rng(hk.transform(_double_critic))
    policy = hk.without_apply_rng(hk.transform(_policy))
    # Create dummy observations and actions to create network parameters.
    dummy_action = utils.zeros_like(spec.actions)
    dummy_obs = utils.zeros_like(spec.observations)
    dummy_action = utils.add_batch_dim(dummy_action)
    dummy_obs = utils.add_batch_dim(dummy_obs)

    return {
        "policy": networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs), policy.apply
        ),
        "critic": networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply
        ),
        "sample": _sample_fn,
    }
