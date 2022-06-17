"""DrQ-v2 builder"""
from typing import Callable, Iterator, List, Optional

from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import optax
import reverb
from reverb import rate_limiters

from magi.agents.drq_v2 import acting as acting_lib
from magi.agents.drq_v2 import config as drq_v2_config
from magi.agents.drq_v2 import learning as learning_lib
from magi.agents.drq_v2 import networks as drq_v2_networks


class DrQV2Builder(builders.ActorLearnerBuilder):
  """DrQ-v2 Builder."""

  def __init__(
      self,
      config: drq_v2_config.DrQV2Config,
      logger_fn: Callable[[], loggers.Logger],
  ):
    self._config = config
    self._logger_fn = logger_fn

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
  ) -> List[reverb.Table]:
    samples_per_insert_tolerance = (
        self._config.samples_per_insert_tolerance_rate *
        self._config.samples_per_insert)
    error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
    limiter = rate_limiters.SampleToInsertRatio(
        min_size_to_sample=self._config.min_replay_size,
        samples_per_insert=self._config.samples_per_insert,
        error_buffer=error_buffer,
    )
    replay_table = reverb.Table(
        name=self._config.replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self._config.max_replay_size,
        rate_limiter=limiter,
        signature=adders_reverb.NStepTransitionAdder.signature(
            environment_spec=environment_spec),
    )
    return [replay_table]

  def make_dataset_iterator(
      self,
      replay_client: reverb.Client,
  ) -> Iterator[reverb.ReplaySample]:
    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=self._config.batch_size,
        prefetch_size=self._config.prefetch_size,
        transition_adder=True,
    )
    return dataset.as_numpy_iterator()

  def make_adder(self, replay_client: reverb.Client) -> Optional[adders.Adder]:

    return adders_reverb.NStepTransitionAdder(
        client=replay_client,
        n_step=self._config.n_step,
        discount=self._config.discount,
    )

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy_network: drq_v2_networks.DrQV2PolicyNetwork,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None,
  ) -> core.Actor:
    assert variable_source is not None
    device = None
    variable_client = variable_utils.VariableClient(
        variable_source, 'policy', device=device)
    variable_client.update_and_wait()

    return acting_lib.DrQV2Actor(
        policy_network,
        random_key,
        variable_client=variable_client,
        adder=adder,
        backend=device,
    )

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: drq_v2_networks.DrQV2Networks,
      dataset: Iterator[reverb.ReplaySample],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:

    del replay_client
    config = self._config
    critic_optimizer = optax.adam(config.learning_rate)
    policy_optimizer = optax.adam(config.learning_rate)
    encoder_optimizer = optax.adam(config.learning_rate)

    sigma_start, sigma_end, sigma_schedule_steps = config.sigma
    observations_per_step = int(config.batch_size / config.samples_per_insert)
    if hasattr(config, 'min_observations'):
      min_observations = config.min_observations
    else:
      min_observations = config.min_replay_size
    # Compute the schedule for the learner
    # Learner only starts updating after min_observations number of steps
    sigma_schedule = lambda step: optax.linear_schedule(  # noqa
        sigma_start, sigma_end, sigma_schedule_steps)((step + max(
            min_observations, config.batch_size)) * observations_per_step)

    return learning_lib.DrQV2Learner(
        random_key=random_key,
        dataset=dataset,
        networks=networks,
        sigma_schedule=sigma_schedule,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        encoder_optimizer=encoder_optimizer,
        augmentation=config.augmentation,
        critic_soft_update_rate=config.critic_q_soft_update_rate,
        discount=config.discount,
        noise_clip=config.noise_clip,
        logger=self._logger_fn(),
        counter=counter,
    )
