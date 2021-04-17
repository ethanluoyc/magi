import haiku as hk
import jax.numpy as jnp
import numpy as np
import tensorflow_probability
from jax import nn

hk_init = hk.initializers
tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions

Initializer = hk.initializers.Initializer

class MLP(hk.Module):
    def __init__(
        self,
        output_dim,
        hidden_units,
        hidden_activation=nn.relu,
        output_activation=None,
        hidden_scale=1.0,
        output_scale=1.0,
    ):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_kwargs = {"w_init": hk.initializers.Orthogonal(scale=hidden_scale)}
        self.output_kwargs = {"w_init": hk.initializers.Orthogonal(scale=output_scale)}

    def __call__(self, x):
        for i, unit in enumerate(self.hidden_units):
            x = hk.Linear(unit, **self.hidden_kwargs)(x)
            x = self.hidden_activation(x)
        x = hk.Linear(self.output_dim, **self.output_kwargs)(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

class GaussianPolicy(hk.Module):
    """
    Policy for SAC.
    """

    def __init__(
        self,
        action_size,
        hidden_units=(256, 256),
        log_std_min=-20.0,
        log_std_max=2.0,
        clip_log_std=True,
        name=None
    ):
        super().__init__(name=name)
        self.hidden_units = hidden_units
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.clip_log_std = clip_log_std
        self.action_size = action_size

    def __call__(self, x):
        x = MLP(
            2 * self.action_size,
            self.hidden_units,
            hidden_activation=nn.relu,
        )(x)
        mean, log_std = jnp.split(x, 2, axis=-1)
        if self.clip_log_std:
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        else:
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (jnp.tanh(log_std) + 1.0)

        # We cannot use MultivariateDiagNormal here as it does not work with the Tanh bijector.
        # TODO(yl): Consider using the distributional heads in newer versions of acme.
        action_tp1_base_dist = tfd.Normal(loc=mean, scale=jnp.exp(log_std))
        action_dist = tfd.TransformedDistribution(action_tp1_base_dist, tfp.bijectors.Tanh())
        return tfd.Independent(action_dist, 1)

class DoubleCritic(hk.Module):
    def __init__(
        self,
        hidden_units=(256, 256),
        name=None
    ):
        super().__init__(name=name)
        self.num_critics = 2
        self.hidden_units = hidden_units

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                1,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )(x)

        x = jnp.concatenate([s, a], axis=1)
        # Return list even if num_critics == 1 for simple implementation.
        return [_fn(x).squeeze(-1) for _ in range(self.num_critics)]
