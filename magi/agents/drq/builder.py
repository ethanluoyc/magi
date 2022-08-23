"""DrQ agent builder."""
from typing import Any, Dict, Iterator, List, Optional

from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import optax
import reverb
from reverb import rate_limiters

from magi.agents.drq import config as drq_config
from magi.agents.drq import learning as learning_lib

DrQNetworks = Dict[str, Any]


class DrQBuilder(builders.ActorLearnerBuilder):
  """Soft Actor-Critic agent specification"""

  def __init__(
      self,
      config: drq_config.DrQConfig,
  ):
    self._config = config

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: actor_core.Policy,
  ) -> List[reverb.Table]:
    del policy
    replay_table = reverb.Table(
        name=self._config.replay_table_name,
        # TODO(yl): support prioritized sampling in SAC
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self._config.max_replay_size,
        rate_limiter=rate_limiters.MinSize(self._config.min_replay_size),
        signature=adders_reverb.NStepTransitionAdder.signature(
            environment_spec=environment_spec),
    )
    return [replay_table]

  def make_dataset_iterator(
      self,
      replay_client: reverb.Client,
  ) -> Iterator[reverb.ReplaySample]:
    """Create a dataset iterator to use for learning/updating the agent."""
    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=self._config.batch_size,
        prefetch_size=self._config.prefetch_size,
        transition_adder=True,
    )
    return dataset.as_numpy_iterator()

  def make_adder(
      self,
      replay_client: reverb.Client,
      environment_spec: Optional[specs.EnvironmentSpec],
      policy: Optional[actor_core.FeedForwardPolicy],
  ) -> Optional[adders.Adder]:
    # TODO(yl): support multi step transitions
    del environment_spec, policy
    return adders_reverb.NStepTransitionAdder(
        client=replay_client, n_step=1, discount=self._config.discount)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: actor_core.FeedForwardPolicy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> core.Actor:
    del environment_spec
    assert variable_source is not None
    variable_client = variable_utils.VariableClient(variable_source, 'policy')
    variable_client.update_and_wait()

    return actors.GenericActor(
        actor_core.batched_feed_forward_to_actor_core(policy),
        random_key,
        variable_client=variable_client,
        adder=adder,
    )

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: DrQNetworks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del replay_client
    del environment_spec
    config = self._config
    critic_optimizer = optax.adam(config.critic_learning_rate)
    actor_optimizer = optax.adam(config.actor_learning_rate)
    temperature_optimizer = optax.adam(
        config.temperature_learning_rate, b1=config.temperature_adam_b1)

    return learning_lib.DrQLearner(
        random_key=random_key,
        dataset=dataset,
        policy_network=networks['actor'],
        critic_network=networks['critic'],
        encoder_network=networks['encoder'],
        policy_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        augmentation=config.augmentation,
        target_entropy=config.target_entropy,
        temperature_optimizer=temperature_optimizer,
        init_temperature=config.init_temperature,
        actor_update_frequency=config.actor_update_frequency,
        critic_target_update_frequency=config.critic_target_update_frequency,
        critic_soft_update_rate=config.critic_q_soft_update_rate,
        discount=config.discount,
        logger=logger_fn('learner'),
        counter=counter,
    )
