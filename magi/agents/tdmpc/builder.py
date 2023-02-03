import dataclasses
from typing import Iterator, List, Optional, Tuple

from acme import adders
from acme import datasets
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax
import reverb

from magi.agents.tdmpc import acting
from magi.agents.tdmpc import learning
from magi.agents.tdmpc import networks as tdmpc_networks

TDMPCNetworks = tdmpc_networks.TDMPCNetworks
TDMPCPolicy = Tuple[tdmpc_networks.TDMPCNetworks, bool]


@dataclasses.dataclass
class TDMPCConfig:
  std_schedule: optax.Schedule
  horizon_schedule: optax.Schedule
  optimizer: optax.GradientTransformation
  batch_size: int = 512
  samples_per_insert: float = 512.0
  samples_per_insert_tolerance_rate: float = 0.1
  max_replay_size: int = int(1e6)
  variable_update_period: int = 1
  per_alpha: float = 0.6
  per_beta: float = 0.4
  discount: float = 0.99
  num_samples: float = 512
  min_std: float = 0.05
  temperature: float = 0.5
  momentum: float = 0.1
  num_elites: int = 64
  iterations: int = 6
  tau: float = 0.01
  seed_steps: int = 5000
  mixture_coef: float = 0.05
  horizon: int = 5
  consistency_coef: float = 2
  reward_coef: float = 0.5
  value_coef: float = 0.1
  rho: float = 0.5


class TDMPCBuilder(builders.ActorLearnerBuilder):

  def __init__(self, config: TDMPCConfig):
    self._config = config

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: Optional[TDMPCPolicy] = None,
  ) -> List[reverb.Table]:
    del policy
    min_replay_size = self._config.seed_steps
    samples_per_insert_tolerance = (
        self._config.samples_per_insert_tolerance_rate *
        self._config.samples_per_insert)
    error_buffer = min_replay_size * samples_per_insert_tolerance
    limiter = reverb.rate_limiters.SampleToInsertRatio(
        min_size_to_sample=min_replay_size,
        samples_per_insert=self._config.samples_per_insert,
        error_buffer=error_buffer,
    )

    return [
        reverb.Table(
            name=adders_reverb.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Prioritized(self._config.per_alpha),
            remover=reverb.selectors.Fifo(),
            rate_limiter=limiter,
            max_size=self._config.max_replay_size,
            signature=adders_reverb.SequenceAdder.signature(
                environment_spec,
                sequence_length=self._config.horizon + 1,
            ),
        )
    ]

  def make_adder(
      self,
      replay_client: reverb.Client,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      policy: Optional[TDMPCPolicy] = None,
  ) -> Optional[adders.Adder]:
    del policy, environment_spec
    return adders_reverb.SequenceAdder(
        replay_client,
        sequence_length=self._config.horizon + 1,
        period=1,
        end_of_episode_behavior=adders_reverb.EndBehavior.WRITE,
        max_in_flight_items=1,
    )

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: TDMPCPolicy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: learning.TDMPCLearner = None,
      adder: Optional[adders.Adder] = None,
  ) -> acting.TDMPCActor:
    networks, evaluation = policy
    variable_client = variable_utils.VariableClient(
        variable_source,
        "policy",
        update_period=self._config.variable_update_period,
    )
    return acting.TDMPCActor(
        variable_client,
        environment_spec,
        networks,
        random_key,
        std_schedule=self._config.std_schedule,
        horizon_schedule=self._config.horizon_schedule,
        discount=self._config.discount,
        num_samples=self._config.num_samples,
        min_std=self._config.min_std,
        temperature=self._config.temperature,
        momentum=self._config.momentum,
        num_elites=self._config.num_elites,
        iterations=self._config.iterations,
        seed_steps=self._config.seed_steps,
        mixture_coef=self._config.mixture_coef,
        horizon=self._config.horizon,
        adder=adder,
        evaluation=evaluation,
    )

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: TDMPCNetworks,
      dataset: Iterator[learning.TDMPCReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> learning.TDMPCLearner:

    loss_scale = learning.LossScalesConfig(
        consistency=self._config.consistency_coef,
        reward=self._config.reward_coef,
        value=self._config.value_coef,
    )

    return learning.TDMPCLearner(
        spec=environment_spec,
        networks=networks,
        random_key=random_key,
        iterator=dataset,
        replay_client=replay_client,
        optimizer=self._config.optimizer,
        per_beta=self._config.per_beta,
        discount=self._config.discount,
        min_std=self._config.min_std,
        tau=self._config.tau,
        loss_scale=loss_scale,
        rho=self._config.rho,
        counter=counter,
        logger=logger_fn("learner"),
    )

  def make_dataset_iterator(
      self,
      replay_client: reverb.Client) -> Iterator[learning.TDMPCReplaySample]:
    dataset = datasets.make_reverb_dataset(
        server_address=replay_client.server_address,
        batch_size=self._config.batch_size,
        num_parallel_calls=4,
    )
    return jax_utils.device_put(
        dataset.as_numpy_iterator(),
        split_fn=jax_utils.keep_key_on_host,
        device=jax.devices()[0],
    )

  def make_policy(
      self,
      networks: TDMPCNetworks,
      environment_spec: specs.EnvironmentSpec,
      evaluation: bool = False,
  ) -> TDMPCPolicy:
    del environment_spec
    return networks, evaluation
