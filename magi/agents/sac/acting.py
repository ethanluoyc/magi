from typing import Optional

from acme import adders
from acme import core
from acme.jax import utils
from acme.jax import variable_utils
import dm_env
import haiku as hk
import jax


class SACActor(core.Actor):
    """A SAC actor."""

    def __init__(
        self,
        forward_fn,
        key,
        is_eval=False,
        variable_client: Optional[variable_utils.VariableClient] = None,
        adder: Optional[adders.Adder] = None,
    ):

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._key = key

        @jax.jit
        def forward(params, key, observation):
            key, subkey = jax.random.split(key)
            observation = utils.add_batch_dim(observation)
            dist = forward_fn(params, observation)
            if is_eval:
                action = dist.mode()
            else:
                action = dist.sample(seed=subkey)
            return utils.squeeze_batch_dim(action), key

        self._forward = forward
        # Make sure not to use a random policy after checkpoint restoration by
        # assigning variables before running the environment loop.
        if self._variable_client is not None:
            self._variable_client.update_and_wait()

    def select_action(self, observation):
        # Forward.
        action, self._key = self._forward(self._params, self._key, observation)
        action = utils.to_numpy(action)
        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder is not None:
            self._adder.add_first(timestep)

    def observe(
        self,
        action,
        next_timestep: dm_env.TimeStep,
    ):
        if self._adder is not None:
            self._adder.add(action, next_timestep)

    def update(self, wait: bool = True):  # not the default wait = False
        if self._variable_client is not None:
            self._variable_client.update(wait=wait)

    @property
    def _params(self) -> Optional[hk.Params]:
        if self._variable_client is None:
            # If self._variable_client is None then we assume self._forward  does not
            # use the parameters it is passed and just return None.
            return None
        return self._variable_client.params
