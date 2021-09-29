from typing import Dict, NamedTuple, Optional, Sequence, Union

from acme import specs
from acme.agents.jax import actors
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class DiscreteValuedHeadOutput(NamedTuple):
    logits: jnp.ndarray
    values: jnp.ndarray


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
        self._values = jnp.linspace(vmin, vmax, num_atoms, axis=-1)
        self._distributional_layer = hk.Linear(
            self._values.shape[0], w_init=w_init, b_init=b_init
        )

    def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
        logits = self._distributional_layer(inputs)
        logits = jnp.reshape(
            logits,
            # Use numpy here since we are computing the shapes statically
            np.concatenate(
                [logits.shape[:1], self._values.shape], axis=0  # batch size
            ),
        )
        values = self._values.astype(logits.dtype)

        return DiscreteValuedHeadOutput(logits=logits, values=values)


class MultivariateNormalDiagHead(hk.Module):
    """Module that produces a multivariate normal distribution using tfd.Independent or
    tfd.MultivariateNormalDiag.
    """

    def __init__(
        self,
        num_dimensions: int,
        init_scale: float = 0.3,
        min_scale: float = 1e-6,
        tanh_mean: bool = False,
        fixed_scale: bool = False,
        use_tfd_independent: bool = False,
        w_init: hk.initializers.Initializer = hk.initializers.VarianceScaling(1e-4),
        b_init: hk.initializers.Initializer = hk.initializers.Constant(0),
    ):
        """Initialization.

        Args:
          num_dimensions: Number of dimensions of MVN distribution.
          init_scale: Initial standard deviation.
          min_scale: Minimum standard deviation.
          tanh_mean: Whether to transform the mean (via tanh) before passing it to
            the distribution.
          fixed_scale: Whether to use a fixed variance.
          use_tfd_independent: Whether to use tfd.Independent or
            tfd.MultivariateNormalDiag class
          w_init: Initialization for linear layer weights.
          b_init: Initialization for linear layer biases.
        """
        super().__init__(name="MultivariateNormalDiagHead")
        self._init_scale = init_scale
        self._min_scale = min_scale
        self._tanh_mean = tanh_mean
        self._mean_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        self._fixed_scale = fixed_scale

        if not fixed_scale:
            self._scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        self._use_tfd_independent = use_tfd_independent

    def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
        zero = 0
        mean = self._mean_layer(inputs)

        if self._fixed_scale:
            scale = jnp.ones_like(mean) * self._init_scale
        else:
            scale = jax.nn.softplus(self._scale_layer(inputs))
            scale *= self._init_scale / jax.nn.softplus(zero)
            scale += self._min_scale

        # Maybe transform the mean.
        if self._tanh_mean:
            mean = jnp.tanh(mean)

        if self._use_tfd_independent:
            dist = tfd.Independent(tfd.Normal(loc=mean, scale=scale))
        else:
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=scale)

        return dist


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
