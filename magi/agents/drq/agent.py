"""Data-regularized Q (DrQ) agent."""
from typing import Any, Dict, Optional

from acme import core
from acme import specs
from acme.utils import counting
from acme.utils import loggers
import dm_env
import jax
import reverb

from magi.agents.drq import builder
from magi.agents.drq import config as drq_config
from magi.agents.drq import networks as drq_networks


class DrQAgent(core.Actor, core.VariableSource):
  """Data-regularized Q agent."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      networks: Dict[str, Any],
      config: drq_config.DrQConfig,
      seed: int,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):
    # Setup reverb
    self.builder = builder.DrQBuilder(config)
    learner_key, actor_key = jax.random.split(jax.random.PRNGKey(seed))
    self._num_observations = 0
    self._min_observations = config.min_replay_size

    policy = drq_networks.apply_policy_sample(networks, eval_mode=False)

    replay_tables = self.builder.make_replay_tables(environment_spec, policy)
    replay_server = reverb.Server(replay_tables, port=None)
    self._server = replay_server

    # The adder is used to insert observations into replay.
    # discount is 1.0 as we are multiplying gamma during learner step
    address = f'localhost:{self._server.port}'
    replay_client = reverb.Client(address)
    # The dataset provides an interface to sample from replay.
    dataset = self.builder.make_dataset_iterator(replay_client)

    self._learner = self.builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=dataset,
        environment_spec=environment_spec,
        logger_fn=lambda _, steps_key=None, task=None: logger,
        counter=counter)

    adder = self.builder.make_adder(replay_client, environment_spec, policy)
    self._actor = self.builder.make_actor(
        actor_key,
        policy,
        environment_spec,
        adder=adder,
        variable_source=self._learner,
    )

  def select_action(self, observation):
    return self._actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    return self._actor.observe_first(timestep)

  def observe(self, action, next_timestep: dm_env.TimeStep):
    self._num_observations += 1
    self._actor.observe(action, next_timestep)

  def _should_update(self):
    return self._num_observations >= self._min_observations

  def update(self, wait: bool = False):
    if self._should_update():
      self._learner.step()
      self._actor.update(wait=wait)

  def get_variables(self, names):
    return self._learner.get_variables(names)
