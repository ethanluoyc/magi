import haiku as hk
import jax
import jax.numpy as jnp

class MLP(hk.Module):
    def __init__(
        self,
        output_dim,
        hidden_units,
        hidden_activation=jax.nn.relu,
        output_activation=None,
        hidden_scale=1.0,
        output_scale=1.0,
        name=None
    ):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_kwargs = {"w_init": hk.initializers.Orthogonal(scale=hidden_scale)}
        self.output_kwargs = {"w_init": hk.initializers.Orthogonal(scale=output_scale)}

    def __call__(self, x):
        for unit in self.hidden_units:
            x = hk.Linear(unit, **self.hidden_kwargs)(x)
            x = self.hidden_activation(x)
        x = hk.Linear(self.output_dim, **self.output_kwargs)(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

class StateDependentGaussianPolicy(hk.Module):
    """
    Policy for SAC.
    """

    def __init__(
        self,
        action_spec,
        hidden_units=(256, 256),
        log_std_min=-20.0,
        log_std_max=2.0,
        clip_log_std=True,
        name=None
    ):
        super().__init__(name=name)
        self.action_spec = action_spec
        self.hidden_units = hidden_units
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.clip_log_std = clip_log_std

    def __call__(self, x):
        x = MLP(
            2 * self.action_spec.shape[0],
            self.hidden_units,
            hidden_activation=jax.nn.relu,
        )(x)
        mean, log_std = jnp.split(x, 2, axis=-1)
        if self.clip_log_std:
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

class ContinuousQFunction(hk.Module):
    """
    Critic for DDPG, TD3 and SAC.
    """

    def __init__(
        self,
        hidden_units=(256, 256),
        name=None
    ):
        super(ContinuousQFunction, self).__init__(name=name)
        self.hidden_units = hidden_units

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                1,
                self.hidden_units,
                hidden_activation=jax.nn.relu,
            )(x)
        x = jnp.concatenate([s, a], axis=-1)
        return jnp.squeeze(_fn(x), axis=-1)
