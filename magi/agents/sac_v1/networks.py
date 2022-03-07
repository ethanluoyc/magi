from typing import Sequence

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256),
    critic_layer_sizes: Sequence[int] = (256, 256),
    value_layer_sizes: Sequence[int] = (256, 256),
):
    num_dimensions = onp.prod(spec.actions.shape, dtype=int)

    def _actor_fn(obs):
        network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(policy_layer_sizes),
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                    activate_final=True,
                ),
                networks_lib.NormalTanhDistribution(num_dimensions),
            ]
        )
        return network(obs)

    def _critic_fn(obs, action):
        network1 = hk.nets.MLP(
            list(critic_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
            activation=jax.nn.relu,
            activate_final=False,
        )
        network2 = hk.nets.MLP(
            list(critic_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
            activation=jax.nn.relu,
            activate_final=False,
        )
        input_ = jnp.concatenate([obs, action], axis=-1)
        value1 = network1(input_)
        value2 = network2(input_)
        return jnp.squeeze(value1, axis=-1), jnp.squeeze(value2, axis=-1)

    def _value_fn(obs):
        network = hk.nets.MLP(
            list(value_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
            activation=jax.nn.relu,
            activate_final=False,
        )
        value = network(obs)
        return jnp.squeeze(value, axis=-1)

    dummy_action = jax_utils.zeros_like(spec.actions)
    dummy_obs = jax_utils.zeros_like(spec.observations)
    dummy_action = jax_utils.add_batch_dim(dummy_action)
    dummy_obs = jax_utils.add_batch_dim(dummy_obs)

    policy = hk.without_apply_rng(hk.transform(_actor_fn, apply_rng=True))
    critic = hk.without_apply_rng(hk.transform(_critic_fn, apply_rng=True))
    value = hk.without_apply_rng(hk.transform(_value_fn, apply_rng=True))

    return {
        "policy": networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs), policy.apply
        ),
        "critic": networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply
        ),
        "value": networks_lib.FeedForwardNetwork(
            lambda key: value.init(key, dummy_obs), value.apply
        ),
    }
