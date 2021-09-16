"""Default network architectures for TD3."""
from typing import Dict, Sequence

from acme import specs
from acme.jax import networks
import haiku as hk
import jax.numpy as jnp
import numpy as np


def make_default_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256),
    critic_layer_sizes: Sequence[int] = (256, 256),
) -> Dict[str, hk.Transformed]:
    """Make default networks used by TD3."""
    action_size = np.prod(action_spec.shape, dtype=int)

    def critic(h):
        output = hk.Sequential(
            [
                hk.nets.MLP(critic_layer_sizes, activate_final=True),
                hk.Linear(1),
            ]
        )(h)
        return jnp.squeeze(output, axis=-1)

    def double_critic(obs, a):
        h = jnp.concatenate([obs, a], axis=-1)
        q1 = critic(h)
        q2 = critic(h)
        return q1, q2

    def policy(obs):
        return hk.Sequential(
            [
                hk.nets.MLP(policy_layer_sizes, activate_final=True),
                hk.Linear(action_size),
                networks.TanhToSpec(action_spec),
            ]
        )(obs)

    critic_network = hk.without_apply_rng(hk.transform(double_critic))
    policy_network = hk.without_apply_rng(hk.transform(policy))

    return {"critic": critic_network, "policy": policy_network}
