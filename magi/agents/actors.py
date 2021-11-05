"""Generic actors."""
import functools
from typing import Optional

from acme import adders
from acme import core
from acme import specs
from acme.jax import networks as network_lib
from acme.jax import utils
import dm_env
import jax
import jax.numpy as jnp
import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


class RandomActor(core.Actor):
    """An actor that samples random actions from the uniform distribution."""

    def __init__(
        self,
        action_spec: specs.BoundedArray,
        random_key: network_lib.PRNGKey,
        adder: Optional[adders.Adder] = None,
    ):
        # Store these for later use.
        self._adder = adder
        self._random_key = random_key

        @functools.partial(jax.jit, backend="cpu")
        def forward(key, observation):
            del observation
            key, subkey = jax.random.split(key)
            action_dist = tfd.Uniform(
                low=jnp.broadcast_to(action_spec.minimum, action_spec.shape),
                high=jnp.broadcast_to(action_spec.maximum, action_spec.shape),
            )
            action = action_dist.sample(seed=subkey)
            return action, key

        self._forward_fn = forward

    def select_action(self, observation: network_lib.Observation) -> network_lib.Action:
        action, self._random_key = self._forward_fn(self._random_key, observation)
        return utils.to_numpy(action)

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder is not None:
            self._adder.add_first(timestep)

    def observe(
        self,
        action: network_lib.Action,
        next_timestep: dm_env.TimeStep,
    ):
        if self._adder is not None:
            self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        pass
