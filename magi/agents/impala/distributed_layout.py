"""Distribtued layout for IMPALA."""
from typing import Any, Callable, Dict, Optional

import acme
from acme import specs
from acme.jax import networks as network_lib
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
import dm_env
import haiku as hk
import jax
import launchpad as lp
import reverb

from magi.agents import utils
from magi.agents.impala import builder as builder_lib


class IMPALADistributedLayout:
  """Program definition for IMPALA."""

  def __init__(
      self,
      environment_factory: Callable[[int, bool], dm_env.Environment],
      network_factory: Callable[[specs.BoundedArray], Dict[str, Any]],
      builder: builder_lib.IMPALABuilder,
      num_actors: int = 1,
      num_caches: int = 0,
      environment_spec: specs.EnvironmentSpec = None,
      max_actor_steps: Optional[int] = None,
      log_every: float = 5.0,
      seed: int = 0,
      logger_fn=None,
  ):
    if num_caches > 0:
      raise ValueError('Caching learners not supported in open-source')

    if not environment_spec:
      environment_spec = specs.make_environment_spec(
          environment_factory(seed, False))

    if not logger_fn:
      logger_fn = loggers.make_default_logger

    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._environment_spec = environment_spec
    self._rng = hk.PRNGSequence(seed)
    self._num_actors = num_actors
    self._num_caches = num_caches
    self._max_actor_steps = max_actor_steps
    self._log_every = log_every
    self._logger_fn = logger_fn
    self._builder = builder

  def replay(self):
    """The replay storage."""
    return self._builder.make_replay_tables(
        environment_spec=self._environment_spec,)

  def counter(self):
    return counting.Counter()

  def coordinator(self, counter: counting.Counter):
    return lp_utils.StepsLimiter(counter, self._max_actor_steps)

  def learner(
      self,
      random_key: network_lib.PRNGKey,
      replay: reverb.Client,
      counter: counting.Counter,
      logger: Optional[loggers.Logger] = None,
  ):
    """The Learning part of the agent."""
    network_dict = self._network_factory(self._environment_spec.actions)
    dataset = self._builder.make_dataset_iterator(replay)
    counter = counting.Counter(counter, 'learner')
    logger = logger or self._logger_fn(
        'learner',
        time_delta=self._log_every,
        save_data=True,
        asynchronous=True,
        steps_key='learner_steps',
    )

    return self._builder.make_learner(
        self._environment_spec,
        network_dict,
        dataset=dataset,
        random_key=random_key,
        counter=counter,
        logger=logger,
    )

  def actor(
      self,
      random_key: network_lib.PRNGKey,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
      logger: Optional[loggers.Logger] = None,
  ) -> acme.EnvironmentLoop:
    """The actor process."""
    network_dict = self._network_factory(self._environment_spec.actions)
    # Create the agent.
    env_key, actor_key = jax.random.split(random_key)
    actor = self._builder.make_actor(
        policy_network=network_dict,
        adder=self._builder.make_adder(replay),
        random_key=actor_key,
        variable_source=variable_source,
    )

    # Create the environment.
    environment = self._environment_factory(utils.rand_seed(env_key), False)

    # Create logger and counter; actors will not spam bigtable.
    counter = counting.Counter(counter, 'actor')
    logger = logger or self._logger_fn(
        'actor',
        save_data=False,
        time_delta=self._log_every,
        steps_key='actor_steps',
    )

    # Create the loop to connect environment and agent.
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def evaluator(
      self,
      random_key: network_lib.PRNGKey,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
      logger: Optional[loggers.Logger] = None,
  ):
    """The evaluation process."""
    # Create the behavior policy.
    network_dict = self._network_factory(self._environment_spec.actions)
    actor_key, env_key = jax.random.split(random_key)
    actor = self._builder.make_actor(
        policy_network=network_dict,
        random_key=actor_key,
        variable_source=variable_source,
    )

    # Make the environment.
    environment = self._environment_factory(utils.rand_seed(env_key), True)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = logger or self._logger_fn(
        'evaluator',
        time_delta=self._log_every,
        steps_key='evaluator_steps',
    )

    # Create the run loop and return it.
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def build(self, name='impala'):
    """Build the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('replay'):
      replay = program.add_node(lp.ReverbNode(self.replay))

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))

    if self._max_actor_steps:
      with program.group('coordinator'):
        _ = program.add_node(lp.CourierNode(self.coordinator, counter))

    with program.group('learner'):
      learner = program.add_node(
          lp.CourierNode(self.learner, next(self._rng), replay, counter))

    with program.group('evaluator'):
      program.add_node(
          lp.CourierNode(self.evaluator, next(self._rng), learner, counter))

    if not self._num_caches:
      # Use our learner as a single variable source.
      sources = [learner]
    else:
      with program.group('cacher'):
        # Create a set of learner caches.
        sources = []
        for _ in range(self._num_caches):
          cacher = program.add_node(
              lp.CacherNode(
                  learner, refresh_interval_ms=2000, stale_after_ms=4000))
          sources.append(cacher)

    with program.group('actor'):
      # Add actors which pull round-robin from our variable sources.
      for actor_id in range(self._num_actors):
        source = sources[actor_id % len(sources)]
        program.add_node(
            lp.CourierNode(self.actor, next(self._rng), replay, source,
                           counter))

    return program
