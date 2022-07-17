"""Distributed base layouts
This is a fork of

  https://github.com/deepmind/acme/blob/master/acme/jax/layouts/distributed_layout.py

with some modifications and simplification to work with our setup. Eventually,
the goal is to directly use the layout definition in Acme when if gives enough
flexibility
"""

from typing import Any, Callable, Optional, Sequence

import acme
from acme import core
from acme import environment_loop
from acme import specs
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
import dm_env
import jax
import launchpad as lp
import reverb
import tensorflow as tf

Networks = Any
NetworkFactory = Callable[[specs.EnvironmentSpec], Networks]
PolicyFactory = Callable[[Networks], Any]
# environment factory takes arguments seed, testing
# This deviates from the original implementation to provide determinism
EnvironmentFactory = Callable[[int, bool], dm_env.Environment]
EvaluatorFactory = Callable[
    [types.PRNGKey, core.VariableSource, counting.Counter], core.Worker]


def _random_key_to_seed(key: types.PRNGKey) -> int:
  return int(jax.random.randint(key, (), minval=0, maxval=2**31 - 1))


def default_evaluator(
    environment_factory: EnvironmentFactory,
    network_factory: NetworkFactory,
    builder: builders.GenericActorLearnerBuilder,
    policy_factory: PolicyFactory,
    logger_fn=loggers.make_default_logger,
):

  def evaluator(
      random_key: networks_lib.PRNGKey,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ):
    """The evaluation process."""
    # Create the behavior policy.
    actor_key, env_key = jax.random.split(random_key)
    # Make the environment.
    environment = environment_factory(_random_key_to_seed(env_key), True)
    environment_spec = specs.make_environment_spec(environment)

    # Make evaluator network
    agent_networks = network_factory(environment_spec)

    # Make actor
    actor = builder.make_actor(
        actor_key,
        policy_factory(agent_networks),
        environment_spec,
        variable_source=variable_source,
    )

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = logger_fn('evaluator', save_data=True, steps_key='actor_steps')
    # Create the run loop and return it.
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  return evaluator


class DistributedLayout:
  """Base layouts for distributed agents.
  TODO(yl): implement checkpointing.
  """

  def __init__(
      self,
      seed: int,
      environment_factory: EnvironmentFactory,
      network_factory: NetworkFactory,
      policy_factory: PolicyFactory,
      builder: builders.ActorLearnerBuilder,
      num_actors: int,
      *,
      environment_spec: specs.EnvironmentSpec = None,
      evaluator_factories: Sequence[EvaluatorFactory] = (),
      max_actor_steps: Optional[int] = None,
      prefetch_size: int = 1,
      prefetch_to_device: bool = True,
      # TODO(yl): we can probably remove this in favor of
      # configuring logging in subclasses
      log_every: float = 5.0,
      multithreading_colocate_learner_and_reverb: bool = False,
      logger_fn=loggers.make_default_logger,
  ):

    self._seed = seed
    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._policy_factory = policy_factory
    self._builder = builder
    self._environment_spec = environment_spec
    self._num_actors = num_actors
    self._evaluator_factories = evaluator_factories
    self._max_actor_steps = max_actor_steps
    self._prefetch_size = prefetch_size
    self._prefetch_to_device = prefetch_to_device
    self._log_every = log_every
    self._multithreading_colocate_learner_and_reverb = (
        multithreading_colocate_learner_and_reverb)
    self._logger_fn = logger_fn

  def replay(self):
    """The replay storage."""
    # TODO(yl) this is a hack to ensure that TF does not use GPU
    # Find a way to do this outside layouts
    tf.config.experimental.set_visible_devices([], 'GPU')
    environment_spec = self._environment_spec or specs.make_environment_spec(
        self._environment_factory(self._seed, True))
    agent_networks = self._network_factory(environment_spec)
    policy = self._policy_factory(agent_networks)
    return self._builder.make_replay_tables(environment_spec, policy)

  def counter(self):
    """The global counter."""
    return counting.Counter()

  def coordinator(self, counter: counting.Counter):
    """Coordinator for stopping the launchpad program."""
    return lp_utils.StepsLimiter(counter, self._max_actor_steps)

  def learner(
      self,
      random_key: networks_lib.PRNGKey,
      replay: reverb.Client,
      counter: counting.Counter,
  ):
    """The learning part of the agent."""
    # TODO(yl) this is a hack to ensure that TF does not use GPU
    # Find a way to do this outside layouts
    tf.config.experimental.set_visible_devices([], 'GPU')

    environment_spec = self._environment_spec or specs.make_environment_spec(
        self._environment_factory(self._seed, True))
    networks = self._network_factory(environment_spec)

    iterator = self._builder.make_dataset_iterator(replay)
    if self._prefetch_size > 1:
      device = jax.devices()[0] if self._prefetch_to_device else None
      iterator = utils.prefetch(iterator, self._prefetch_size, device=device)

    counter = counting.Counter(counter, 'learner')
    learner = self._builder.make_learner(
        random_key,
        networks,
        dataset=iterator,
        logger_fn=self._logger_fn,
        environment_spec=environment_spec,
        counter=counter,
    )

    return learner

  def actor(
      self,
      actor_id: int,
      random_key: networks_lib.PRNGKey,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ) -> acme.EnvironmentLoop:
    """The actor process."""
    # Create the agent.
    environment_key, actor_key = jax.random.split(random_key)
    # Create the environment.
    environment = self._environment_factory(
        _random_key_to_seed(environment_key), False)
    environment_spec = specs.make_environment_spec(environment)
    agent_networks = self._network_factory(environment_spec)
    policy = self._policy_factory(agent_networks)
    actor = self._builder.make_actor(
        actor_key,
        policy,
        environment_spec,
        adder=self._builder.make_adder(replay),
        variable_source=variable_source,
    )

    counter = counting.Counter(counter, 'actor')
    logger = self._logger_fn(
        label='actor',
        save_data=actor_id == 0,
        time_delta=self._log_every,
        steps_key='actor_steps',
    )

    return environment_loop.EnvironmentLoop(environment, actor, counter, logger)

  def build(self, name='agent'):
    """Build the distributed agent topology."""
    program = lp.Program(name=name)
    key = jax.random.PRNGKey(self._seed)

    replay_node = lp.ReverbNode(self.replay)
    with program.group('replay'):
      if self._multithreading_colocate_learner_and_reverb:
        replay = replay_node.create_handle()
      else:
        replay = program.add_node(replay_node)

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))

    if self._max_actor_steps:
      with program.group('coordinator'):
        _ = program.add_node(lp.CourierNode(self.coordinator, counter))

    learner_key, key = jax.random.split(key)
    learner_node = lp.CourierNode(self.learner, learner_key, replay, counter)
    with program.group('learner'):
      if self._multithreading_colocate_learner_and_reverb:
        learner = learner_node.create_handle()
        program.add_node(
            lp.MultiThreadingColocation([learner_node, replay_node]))
      else:
        learner = program.add_node(learner_node)

    with program.group('evaluator'):
      for evaluator in self._evaluator_factories:
        evaluator_key, key = jax.random.split(key)
        program.add_node(
            lp.CourierNode(evaluator, evaluator_key, learner, counter))

    with program.group('actor'):
      for actor_id in range(self._num_actors):
        actor_key, key = jax.random.split(key)
        program.add_node(
            lp.CourierNode(self.actor, actor_id, actor_key, replay, learner,
                           counter))

    return program
