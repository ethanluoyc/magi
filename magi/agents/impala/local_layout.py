"""Single-process IMPALA Agent."""
from typing import Optional, Sequence

import acme
from acme import specs
from acme.agents.jax.impala import types
from acme.utils import counting
from acme.utils import loggers
import dm_env
import jax
import numpy as np
import reverb

from magi.agents.impala import builder as builder_lib


class IMPALALocalLayout(acme.Actor, acme.VariableSource):
    """IMPALA Agent."""

    def __init__(
        self,
        networks,
        environment_spec: specs.EnvironmentSpec,
        builder: builder_lib.IMPALABuilder,
        batch_size: int,
        seed: int = 0,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        self._builder = builder
        reverb_queue = self._builder.make_replay_tables(
            environment_spec=environment_spec
        )
        self._server = reverb.Server(reverb_queue)
        self._can_sample = lambda: reverb_queue[0].can_sample(batch_size)
        address = f"localhost:{self._server.port}"
        replay_client = reverb.Client(address)

        key_learner, key_actor = jax.random.split(jax.random.PRNGKey(seed))
        data_iterator = self._builder.make_dataset_iterator(replay_client)
        self._learner = self._builder.make_learner(
            environment_spec,
            networks,
            key_learner,
            data_iterator,
            logger=logger,
            counter=counter,
        )

        # Make the actor.
        self._actor = self._builder.make_actor(
            networks,
            key_actor,
            adder=self._builder.make_adder(replay_client),
            variable_source=self._learner,
        )

    def observe_first(self, timestep: dm_env.TimeStep):
        self._actor.observe_first(timestep)

    def observe(
        self,
        action: types.Action,
        next_timestep: dm_env.TimeStep,
    ):
        self._actor.observe(action, next_timestep)

    def update(self, wait: bool = False):
        should_update_actor = False
        # Run a number of learner steps (usually gradient steps).
        while self._can_sample():
            self._learner.step()
            should_update_actor = True
        if should_update_actor:
            # Update actor weights after learner.
            self._actor.update(wait)

    def select_action(self, observation: np.ndarray) -> int:
        return self._actor.select_action(observation)

    def get_variables(self, names: Sequence[str]):
        return self._learner.get_variables(names)
