"""Data-regularized Q version 2 (DrQ-v2) agent."""
from typing import Optional

import jax
import optax
import reverb
from acme import specs
from acme.agents import agent as agent_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers

from magi.agents.drq_v2 import builder
from magi.agents.drq_v2 import config as drq_v2_config
from magi.agents.drq_v2 import networks as drq_v2_networks


class DrQV2(agent_lib.Agent):
    """Data-regularized Q agent version 2."""

    builder: builder.DrQV2Builder

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        networks: drq_v2_networks.DrQV2Networks,
        config: drq_v2_config.DrQV2Config,
        seed: int,
        device_prefetch: bool = True,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        # This prevents reverb from deadlocking
        min_replay_size = config.min_replay_size
        # Local layout (actually agent.Agent) makes sure that we populate the
        # buffer with min_replay_size initial transitions and that there's no need
        # for tolerance_rate. In order to avoid deadlocks, we disable rate limiting
        # that is configured in DrQV2Builder.make_replay_tables. This is achieved by
        # the following two lines.
        config.samples_per_insert_tolerance_rate = float("inf")
        config.min_replay_size = 1
        config.min_observations = min_replay_size
        self.builder = builder.DrQV2Builder(config)

        random_key = jax.random.PRNGKey(seed)
        learner_key, actor_key = jax.random.split(random_key)
        policy = drq_v2_networks.get_default_behavior_policy(
            networks, environment_spec.actions, optax.linear_schedule(*config.sigma)
        )

        # Setup reverb
        replay_tables = self.builder.make_replay_tables(environment_spec, policy)
        replay_server = reverb.Server(replay_tables, port=None)
        self._server = replay_server

        address = f"localhost:{self._server.port}"
        replay_client = reverb.Client(address)
        # The dataset provides an interface to sample from replay.
        dataset = self.builder.make_dataset_iterator(replay_client)
        if config.prefetch_size is not None and config.prefetch_size > 1:
            device = jax.devices()[0] if device_prefetch else None
            dataset = utils.prefetch(dataset, config.prefetch_size, device)
        learner = self.builder.make_learner(
            random_key=learner_key,
            networks=networks,
            dataset=dataset,
            environment_spec=environment_spec,
            logger_fn=lambda _, steps_key=None, task=None: logger,
            counter=counter,
        )

        adder = self.builder.make_adder(replay_client, environment_spec, policy)
        actor = self.builder.make_actor(
            actor_key,
            policy,
            environment_spec,
            adder=adder,
            variable_source=learner,
        )
        effective_batch_size = config.batch_size
        super().__init__(
            actor,
            learner,
            min_observations=max(config.min_observations, effective_batch_size),
            observations_per_step=float(effective_batch_size)
            / config.samples_per_insert,
        )
