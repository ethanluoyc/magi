"""Actors for PPO."""
from typing import Optional

from acme import core
from acme.agents.jax.impala import types
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp


class PPOActor(core.Actor):
    """An actor which selects actions based on the categorical distribution."""

    def __init__(self, network_fn, variable_client, adder, key):
        self.rng = hk.PRNGSequence(key)
        self._variable_client = variable_client
        self._adder = adder
        network = hk.without_apply_rng(hk.transform(network_fn))
        self._actor = network

        @jax.jit
        def forward_fn(params, key, observation):
            observation = observation[None, ...]
            logits, value = self._actor.apply(params, observation)
            logits = jnp.squeeze(logits, 0)
            value = value.squeeze(0)
            action = jax.random.categorical(key, logits)
            return action, value, logits

        self._forward_fn = forward_fn

        self._prev_logits = None
        self._prev_value = None

    def select_action(self, observation: types.Observation) -> types.Action:
        action, value, logits = self._forward_fn(
            self._params, next(self.rng), observation
        )
        self._prev_logits = logits
        self._prev_value = value
        return action.item()

    def observe_first(self, timestep: dm_env.TimeStep):
        self._adder.add_first(timestep)

    def observe(
        self,
        action: types.Action,
        next_timestep: dm_env.TimeStep,
    ):
        extras = {"value": self._prev_value, "logits": self._prev_logits}
        self._adder.add(action, next_timestep, extras)

    def update(self, wait: bool = True):
        self._variable_client.update(wait)

    @property
    def _params(self) -> Optional[hk.Params]:
        return self._variable_client.params


class GreedyActor(core.Actor):
    """A greedy actor which selects the most probably action."""

    def __init__(self, actor_fn, variable_client, key):
        self.rng = hk.PRNGSequence(key)
        self._variable_client = variable_client
        actor_fn = hk.without_apply_rng(hk.transform(actor_fn))
        self._actor = actor_fn

        @jax.jit
        def forward_fn(params, key, observation):
            del key
            observation = observation[None, ...]
            logits, _ = self._actor.apply(params, observation)
            probs = jax.nn.softmax(logits)
            action = jnp.argmax(probs)
            return action

        self._forward_fn = forward_fn

    def select_action(self, observation: types.Observation) -> types.Action:
        action = self._forward_fn(self._params, next(self.rng), observation)
        return action.item()

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(self, action: types.Action, next_timestep: dm_env.TimeStep):
        pass

    def update(self, wait: bool = True):
        if self._variable_client is not None:
            self._variable_client.update(wait)

    @property
    def _params(self) -> Optional[hk.Params]:
        if self._variable_client is None:
            # If self._variable_client is None then we assume self._forward  does not
            # use the parameters it is passed and just return None.
            return None
        return self._variable_client.params
