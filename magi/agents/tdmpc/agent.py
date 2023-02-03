import sys
import time
from typing import Optional, Sequence, Tuple

from acme import specs
from acme.jax import utils as jax_utils
from acme.utils import loggers
import jax
import optax
import acme
import reverb

from magi.agents.tdmpc import builder as builder_lib
from magi.agents.tdmpc import networks as tdmpc_networks


class TDMPC(acme.Actor, acme.VariableSource):

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      networks: tdmpc_networks.TDMPCNetworks,
      random_key: jax.random.PRNGKeyArray,
      *,
      std_schedule: optax.Schedule,
      horizon_schedule: optax.Schedule,
      optimizer: optax.GradientTransformation,
      batch_size: int = 512,
      samples_per_insert: float = 512.0,
      samples_per_insert_tolerance_rate: float = 0.1,
      max_replay_size: int = int(1e6),
      per_alpha: float = 0.6,
      per_beta: float = 0.4,
      discount: float = 0.99,
      num_samples: float = 512,
      min_std: float = 0.05,
      temperature: float = 0.5,
      momentum: float = 0.1,
      num_elites: int = 64,
      iterations: int = 6,
      tau: float = 0.01,
      seed_steps: int = 5000,
      mixture_coef: float = 0.05,
      horizon: int = 5,
      consistency_coef: float = 2,
      reward_coef: float = 0.5,
      value_coef: float = 0.1,
      rho: float = 0.5,
      logger: Optional[loggers.Logger] = None,
  ):
    tdmpc_config = builder_lib.TDMPCConfig(
        std_schedule=std_schedule,
        horizon_schedule=horizon_schedule,
        optimizer=optimizer,
        batch_size=batch_size,
        samples_per_insert=samples_per_insert,
        samples_per_insert_tolerance_rate=samples_per_insert_tolerance_rate,
        max_replay_size=max_replay_size,
        per_alpha=per_alpha,
        per_beta=per_beta,
        discount=discount,
        num_samples=num_samples,
        min_std=min_std,
        temperature=temperature,
        momentum=momentum,
        num_elites=num_elites,
        iterations=iterations,
        tau=tau,
        seed_steps=seed_steps,
        mixture_coef=mixture_coef,
        horizon=horizon,
        consistency_coef=consistency_coef,
        reward_coef=reward_coef,
        value_coef=value_coef,
        rho=rho,
    )

    builder = builder_lib.TDMPCBuilder(tdmpc_config)
    self.builder = builder

    replay_tables = builder.make_replay_tables(environment_spec=spec)

    self._replay_tables, self._sample_sizes = _disable_insert_blocking(
        replay_tables)

    self._replay_server = reverb.Server(self._replay_tables)
    replay_client = reverb.Client(f"localhost:{self._replay_server.port}")

    dataset = builder.make_dataset_iterator(replay_client)
    dataset = jax_utils.prefetch(dataset)
    self._iterator = dataset

    learner_key, actor_key = jax.random.split(random_key)
    logger_fn = lambda *_: logger
    self._learner = builder.make_learner(
        learner_key,
        networks,
        dataset,
        logger_fn=logger_fn,
        environment_spec=spec,
        replay_client=replay_client,
        counter=None,
    )

    actor_key, eval_actor_key = jax.random.split(actor_key)
    adder = builder.make_adder(replay_client, environment_spec=spec)
    policy = builder.make_policy(
        networks, environment_spec=spec, evaluation=False)
    self._actor = builder.make_actor(
        actor_key, policy, spec, self._learner, adder=adder)
    eval_policy = builder.make_policy(
        networks, environment_spec=spec, evaluation=True)
    self.eval_actor = builder.make_actor(
        eval_actor_key,
        eval_policy,
        environment_spec=spec,
        variable_source=self._learner,
    )
    self._learner_steps = 0

  def save(self):
    return self._learner.save()

  def restore(self, state):
    return self._learner.restore(state)

  def observe_first(self, timestep):
    return self._actor.observe_first(timestep)

  def observe(self, action, next_timestep):
    self._actor.observe(action, next_timestep)

  def select_action(self, obs):
    return self._actor.select_action(obs)

  def _maybe_train(self):
    trained = False
    while True:
      if self._iterator.ready():
        self._learner.step()
        batches = self._iterator.retrieved_elements() - self._learner_steps
        self._learner_steps += 1
        assert batches == 1, (
            "Learner step must retrieve exactly one element from the iterator"
            f" (retrieved {batches}). Otherwise agent can deadlock. Example "
            "cause is that your chosen agent"
            "s Builder has a `make_learner` "
            "factory that prefetches the data but it shouldn"
            "t.")
        trained = True
      else:
        # Wait for the iterator to fetch more data from the table(s) only
        # if there plenty of data to sample from each table.
        for table, sample_size in zip(self._replay_tables, self._sample_sizes):
          if not table.can_sample(sample_size):
            return trained
        # Let iterator's prefetching thread get data from the table(s).
        time.sleep(0.001)

  def update(self, wait: bool = False):
    del wait
    trained = self._maybe_train()
    if trained:
      self._actor.update()

  def get_variables(self, names):
    return self._learner.get_variables(names)


def _disable_insert_blocking(
    tables: Sequence[reverb.Table],
) -> Tuple[Sequence[reverb.Table], Sequence[int]]:
  """Disables blocking of insert operations for a given collection of tables."""
  modified_tables = []
  sample_sizes = []
  for table in tables:
    rate_limiter_info = table.info.rate_limiter_info
    rate_limiter = reverb.rate_limiters.RateLimiter(
        samples_per_insert=rate_limiter_info.samples_per_insert,
        min_size_to_sample=rate_limiter_info.min_size_to_sample,
        min_diff=rate_limiter_info.min_diff,
        max_diff=sys.float_info.max,
    )
    modified_tables.append(table.replace(rate_limiter=rate_limiter))
    # Target the middle of the rate limiter's insert-sample balance window.
    sample_sizes.append(
        max(1, int(
            (rate_limiter_info.max_diff - rate_limiter_info.min_diff) / 2)))
  return modified_tables, sample_sizes
