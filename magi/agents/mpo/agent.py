"""MPO Agent"""
from typing import Optional

from acme import specs
from acme.agents import agent as agent_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import reverb

from magi.agents.mpo import builder as mpo_builder
from magi.agents.mpo import config as mpo_config
from magi.agents.mpo import networks as mpo_networks


class MPO(agent_lib.Agent):
  """MPO agent."""

  builder: mpo_builder.MPOBuilder

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      networks,
      config: mpo_config.MPOConfig,
      seed: int,
      device_prefetch: bool = True,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):
    # This prevents reverb from deadlocking in single-process mode.
    min_replay_size = config.min_replay_size
    # Local layout (actually agent_lib.Agent) makes sure that we populate the
    # buffer with min_replay_size initial transitions and that there's no need
    # for tolerance_rate. In order to avoid deadlocks, we disable rate limiting
    # that is configured in MPOBuilder.make_replay_tables. This is achieved by
    # the following two lines.
    config.samples_per_insert_tolerance_rate = float('inf')
    config.min_replay_size = 1
    config.min_observations = min_replay_size
    self.builder = mpo_builder.MPOBuilder(config, logger_fn=lambda: logger)

    random_key = jax.random.PRNGKey(seed)
    learner_key, actor_key = jax.random.split(random_key)

    # Setup reverb
    replay_tables = self.builder.make_replay_tables(environment_spec)
    replay_server = reverb.Server(replay_tables, port=None)
    self._server = replay_server

    address = f'localhost:{self._server.port}'
    replay_client = reverb.Client(address)
    # The dataset provides an interface to sample from replay.
    dataset = self.builder.make_dataset_iterator(replay_client)
    if config.prefetch_size is not None and config.prefetch_size > 1:
      device = jax.devices()[0] if device_prefetch else None
      dataset = utils.prefetch(dataset, config.prefetch_size, device)
    learner = self.builder.make_learner(
        learner_key, networks, dataset, counter=counter)

    adder = self.builder.make_adder(replay_client)
    actor = self.builder.make_actor(
        actor_key,
        mpo_networks.apply_policy_and_sample(
            networks, environment_spec.actions, eval_mode=False),
        adder=adder,
        variable_source=learner,
    )
    effective_batch_size = config.batch_size
    super().__init__(
        actor,
        learner,
        min_observations=max(config.min_observations, effective_batch_size),
        observations_per_step=float(effective_batch_size) /
        config.samples_per_insert,
    )
