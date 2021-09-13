from acme.jax.networks import continuous
from acme.jax.networks import distributional
import haiku as hk
import jax.numpy as jnp


class Actor(hk.Module):
    """
    Policy for SAC.
    """

    def __init__(
        self,
        action_size,
        hidden_units=(256, 256, 256, 256),
        clip_log_std=True,
        name=None,
    ):
        super().__init__(name=name)
        self.hidden_units = hidden_units
        self.clip_log_std = clip_log_std
        self.action_size = action_size

    def __call__(self, x):
        torso = continuous.LayerNormMLP(
            layer_sizes=self.hidden_units, activate_final=True
        )
        h = torso(x)
        return distributional.NormalTanhDistribution(self.action_size)(h)


class DoubleCritic(hk.Module):
    def __init__(self, hidden_units=(256, 256, 256, 256), name=None):
        super().__init__(name=name)
        self.hidden_units = hidden_units

    def __call__(self, s, a):
        def _fn(x):
            torso = continuous.LayerNormMLP(
                layer_sizes=self.hidden_units, activate_final=True
            )
            head = hk.Linear(1)
            return jnp.squeeze(head(torso(x)), axis=-1)

        x = jnp.concatenate([s, a], axis=-1)
        # Return list even if num_critics == 1 for simple implementation.
        return _fn(x), _fn(x)
