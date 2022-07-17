"""Importance weighted advantage actor-critic (IMPALA) agent implementation."""

from typing import Any, Dict, Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as reverb_adders
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import numpy as np
import optax
import reverb

from magi.agents.impala import acting
from magi.agents.impala import config as impala_config
from magi.agents.impala import learning

IMPALANetworks = Dict[str, Any]


class IMPALABuilder:
  """Builder for IMPALA which constructs individual components of the agent."""

  def __init__(self, config: impala_config.IMPALAConfig, initial_state):
    self._config = config
    self._initial_state = initial_state

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy,
  ) -> List[reverb.Table]:
    del policy
    extra_spec = {
        'core_state':
            self._initial_state,
        'logits':
            np.ones(
                shape=(environment_spec.actions.num_values,), dtype=np.float32),
    }
    signature = reverb_adders.SequenceAdder.signature(environment_spec,
                                                      extra_spec)
    reverb_queue = reverb.Table.queue(
        name=self._config.replay_table_name,
        max_size=self._config.max_queue_size,
        signature=signature,
    )

    return [reverb_queue]

  def make_dataset_iterator(
      self,
      reverb_client: reverb.Client,
  ) -> Iterator[reverb.ReplaySample]:
    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=reverb_client.server_address,
        table=self._config.replay_table_name,
        max_in_flight_samples_per_worker=1,
    )
    dataset = dataset.batch(self._config.batch_size, drop_remainder=True)
    return dataset.as_numpy_iterator()

  def make_adder(
      self,
      replay_client: reverb.Client,
  ) -> adders.Adder:
    adder = reverb_adders.SequenceAdder(
        client=reverb.Client(replay_client.server_address),
        period=self._config.sequence_period,
        sequence_length=self._config.sequence_length,
        pad_end_of_episode=bool(self._config.break_end_of_episode),
        break_end_of_episode=self._config.break_end_of_episode,
    )
    return adder

  def make_actor(
      self,
      policy_network,
      random_key,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None,
  ):
    if variable_source:
      # Create the variable client responsible for keeping the actor up-to-date.
      variable_client = variable_utils.VariableClient(
          client=variable_source,
          key='policy',
          update_period=self._config.sequence_length,
      )
      # Make sure not to use a random policy after checkpoint restoration by
      # assigning variables before running the environment loop.
      variable_client.update_and_wait()

    else:
      variable_client = None

    # Create the actor which defines how we take actions.
    transformed_forward = hk.without_apply_rng(
        hk.transform(policy_network['forward']))
    return acting.IMPALAActor(
        forward_fn=jax.jit(transformed_forward.apply, backend='cpu'),
        initial_state_fn=policy_network['initial_state'],
        rng=hk.PRNGSequence(random_key),
        adder=adder,
        variable_client=variable_client,
    )

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: IMPALANetworks,
      dataset: Iterator[reverb.Sample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ):
    del replay_client

    optimizer = optax.chain(
        optax.clip_by_global_norm(self._config.max_gradient_norm),
        optax.adam(self._config.learning_rate),
    )
    return learning.IMPALALearner(
        obs_spec=environment_spec.observations,
        unroll_fn=networks['unroll'],
        initial_state_fn=networks['initial_state'],
        iterator=dataset,
        random_key=random_key,
        counter=counter,
        logger=logger_fn('learner'),
        optimizer=optimizer,
        discount=self._config.discount,
        entropy_cost=self._config.entropy_cost,
        baseline_cost=self._config.baseline_cost,
        max_abs_reward=self._config.max_abs_reward,
    )
