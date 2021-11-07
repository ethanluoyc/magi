from typing import Optional

from acme import adders
from acme import core
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
import jax

from magi.agents.drq_v2 import networks as drq_v2_networks


class DrQV2Actor(core.Actor):
    def __init__(
        self,
        policy: drq_v2_networks.DrQV2PolicyNetwork,
        random_key: networks_lib.PRNGKey,
        variable_client: Optional[variable_utils.VariableClient],
        adder: Optional[adders.Adder] = None,
        jit: bool = True,
        backend: Optional[str] = "cpu",
        per_episode_update: bool = False,
    ):
        """Initializes a DrQV2Actor.
        Args:
            policy: DrQV2 policy network.
            random_key: Random key.
            variable_client: The variable client to get policy parameters from.
            adder: An adder to add experiences to.
            jit: Whether or not to jit the passed ActorCore's pure functions.
            backend: Which backend to use when jitting the policy.
            per_episode_update: if True, updates variable client params once at the
                beginning of each episode
        """
        self._random_key = random_key
        self._variable_client = variable_client
        self._adder = adder
        self._state = None
        self._num_steps = 0
        self._per_episode_update = per_episode_update

        def init(rng):
            return rng

        def select_action(params, observation, state, step):
            rng = state
            rng1, rng2 = jax.random.split(rng)
            observation = utils.add_batch_dim(observation)
            action = utils.squeeze_batch_dim(policy(params, rng1, observation, step))
            return action, rng2

        def get_extras(unused):
            del unused

        # Unpack ActorCore, jitting if requested.
        if jit:
            self._init = jax.jit(init, backend=backend)
            self._policy = jax.jit(select_action, backend=backend)
        else:
            self._init = init
            self._policy = select_action
            self._get_extras = get_extras
            self._per_episode_update = per_episode_update
        self._get_extras = get_extras

    @property
    def _params(self):
        return self._variable_client.params if self._variable_client else []

    def select_action(self, observation) -> types.NestedArray:
        action, self._state = self._policy(
            self._params, observation, self._state, self._num_steps
        )
        self._num_steps += 1
        return utils.to_numpy(action)

    def observe_first(self, timestep):
        self._random_key, key = jax.random.split(self._random_key)
        self._state = self._init(key)
        if self._adder:
            self._adder.add_first(timestep)
        if self._variable_client and self._per_episode_update:
            self._variable_client.update_and_wait()

    def observe(self, action, next_timestep):
        if self._adder:
            self._adder.add(action, next_timestep, extras=self._get_extras(self._state))

    def update(self, wait: bool = False):
        if self._variable_client and not self._per_episode_update:
            self._variable_client.update(wait)
