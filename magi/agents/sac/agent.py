"""Soft Actor-Critic implementation"""
from typing import Dict, Optional, Sequence

from acme import core
from acme import specs
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
import dm_env
import jax
import reverb

from magi.agents.sac import builder
from magi.agents.sac import config as sac_config


class SACAgent(core.Actor):
    """Single process SAC agent"""

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        networks: Dict[str, networks_lib.FeedForwardNetwork],
        seed: int,
        config: sac_config.SACConfig,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):
        # If the user does not specify a target entropy, infer it from the spec.
        self.builder = builder.SACBuilder(config, logger_fn=lambda: logger)
        learner_key, actor_key = jax.random.split(jax.random.PRNGKey(seed))
        self._num_observations = 0
        self._min_observations = config.min_replay_size

        replay_tables = self.builder.make_replay_tables(environment_spec)
        replay_server = reverb.Server(replay_tables, port=None)
        self._server = replay_server

        # The adder is used to insert observations into replay.
        # discount is 1.0 as we are multiplying gamma during learner step
        address = f"localhost:{self._server.port}"
        replay_client = reverb.Client(address)
        # The dataset provides an interface to sample from replay.
        dataset = self.builder.make_dataset_iterator(replay_client)

        self._learner = self.builder.make_learner(
            learner_key, networks, dataset, counter=counter
        )

        def policy_network(params, key, obs):
            return networks["policy"].apply(params, obs).sample(seed=key)

        adder = self.builder.make_adder(replay_client)
        self._actor = self.builder.make_actor(
            actor_key,
            policy_network,
            adder=adder,
            variable_source=self._learner,
        )

    def select_action(
        self, observation: networks_lib.Observation
    ) -> networks_lib.Action:
        return self._actor.select_action(observation)

    def observe_first(self, timestep: dm_env.TimeStep):
        return self._actor.observe_first(timestep)

    def observe(self, action: networks_lib.Action, next_timestep: dm_env.TimeStep):
        self._num_observations += 1
        self._actor.observe(action, next_timestep)

    def update(self):
        if self._num_observations <= self._min_observations:
            return
        self._learner.step()
        self._actor.update(wait=True)

    def get_variables(self, names: Sequence[str]):
        return self._learner.get_variables(names)
