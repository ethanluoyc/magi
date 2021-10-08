from typing import Dict, NamedTuple, Optional, Sequence, Union

from acme import specs
from acme.agents.jax import actors
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class DiscreteValuedHeadOutput(NamedTuple):
    logits: jnp.ndarray
    atoms: jnp.ndarray


class DiscreteValuedHead(hk.Module):
    """Represents a parameterized discrete valued distribution.
    The returned distribution is essentially a `tfd.Categorical`, but one which
    knows its support and so can compute the mean value.
    """

    def __init__(
        self,
        vmin: Union[float, np.ndarray, jnp.ndarray],
        vmax: Union[float, np.ndarray, jnp.ndarray],
        num_atoms: int,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
    ):
        """Initialization.
        If vmin and vmax have shape S, this will store the category values as a
        Tensor of shape (S*, num_atoms).
        Args:
          vmin: Minimum of the value range
          vmax: Maximum of the value range
          num_atoms: The atom values associated with each bin.
          w_init: Initialization for linear layer weights.
          b_init: Initialization for linear layer biases.
        """
        super().__init__(name="DiscreteValuedHead")
        self._atoms = jnp.linspace(vmin, vmax, num_atoms, axis=-1)
        self._distributional_layer = hk.Linear(
            self._atoms.shape[0], w_init=w_init, b_init=b_init
        )

    def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
        logits = self._distributional_layer(inputs)
        logits = jnp.reshape(
            logits,
            # Use numpy here since we are computing the shapes statically
            np.concatenate([logits.shape[:1], self._atoms.shape], axis=0),  # batch size
        )
        atoms = self._atoms.astype(logits.dtype)

        return DiscreteValuedHeadOutput(logits=logits, atoms=atoms)


def apply_policy_and_sample(
    networks: Dict[str, networks_lib.FeedForwardNetwork],
    eval_mode: bool = False,
) -> actors.FeedForwardPolicy:
    """Returns a function that computes actions."""
    policy_network = networks["policy"]

    def apply_and_sample(params, key, obs):
        action_dist = policy_network.apply(params, obs)
        action = action_dist.mode() if eval_mode else action_dist.sample(seed=key)
        return action

    return apply_and_sample


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (256, 256, 256),
    vmin: float = -150.0,
    vmax: float = 150.0,
    num_atoms: int = 51,
):
    """Creates networks used by the CRR agent."""
    # Get total number of action dimensions from action spec.
    num_dimensions = int(np.prod(spec.actions.shape, dtype=int))

    def _policy_fn(observations):
        # Create the policy network.
        policy_network = hk.Sequential(
            [
                networks_lib.LayerNormMLP(policy_layer_sizes, activate_final=True),
                networks_lib.NormalTanhDistribution(num_dimensions),
            ]
        )
        return policy_network(observations)

    def _critic_fn(observations, actions):
        # Create the critic network.
        critic_network = hk.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
                networks_lib.CriticMultiplexer(),
                networks_lib.LayerNormMLP(critic_layer_sizes, activate_final=True),
                DiscreteValuedHead(vmin, vmax, num_atoms),
            ]
        )
        return critic_network(observations, actions)

    # Create dummy observations and actions to create network parameters.
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
