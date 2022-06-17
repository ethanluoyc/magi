"""TD3 agent implementation"""
from typing import Dict, List, Optional, Sequence

import acme
from acme import specs
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
import jax
import jax.numpy as jnp
import reverb
import tensorflow_probability.substrates.jax as tfp

from magi.agents.td3 import builder as builder_lib
from magi.agents.td3 import config as td3_config
from magi.agents.td3 import networks as td3_networks

tfd = tfp.distributions


class TD3Agent(acme.Actor, acme.VariableSource):
  """Single-process TD3 agent implementation.

    References:
        [1]: Fujimoto, Scott and Hoof, Herke and Meger, David.
             Addressing Function Approximation Error in Actor-Critic Methods.
             In _International Conference on Machine Learning_, 2018.
             https://arxiv.org/abs/1802.09477
    """

  builder: builder_lib.TD3Builder

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      networks: Dict[str, networks_lib.FeedForwardNetwork],
      config: td3_config.TD3Config,
      random_key: jax_types.PRNGKey,
      logger: Optional[loggers.Logger] = None,
      counter: Optional[counting.Counter] = None,
  ):
    min_replay_size = config.min_replay_size
    config.min_replay_size = 1
    self.builder = builder_lib.TD3Builder(config, logger_fn=lambda: logger)
    # Create the replay server and grab its address.
    replay_tables = self.builder.make_replay_tables(environment_spec)
    replay_server = reverb.Server(replay_tables, port=None)
    replay_client = reverb.Client(f'localhost:{replay_server.port}')

    # Create actor, dataset, and learner for generating, storing, and consuming
    # data respectively.
    learner_key, actor_key, self._key = jax.random.split(random_key, 3)

    dataset = self.builder.make_dataset_iterator(replay_client)
    learner = self.builder.make_learner(
        learner_key,
        networks,
        dataset,
        counter=counter,
    )
    adder = self.builder.make_adder(replay_client)
    actor = self.builder.make_actor(
        actor_key,
        td3_networks.apply_policy_sample(networks, eval_mode=False),
        adder=adder,
        variable_source=learner,
    )

    def random_exploration_policy(key, observation):
      del observation
      action_spec = environment_spec.actions
      key, subkey = jax.random.split(key)
      action_dist = tfd.Uniform(
          low=jnp.broadcast_to(action_spec.minimum, action_spec.shape),
          high=jnp.broadcast_to(action_spec.maximum, action_spec.shape),
      )
      action = action_dist.sample(seed=subkey)
      return action, key

    # Internalize agent components
    self._actor = actor
    self._learner = learner
    self._replay_server = replay_server
    self._num_observations = 0
    self._min_observations = min_replay_size
    self._random_exploration_policy = jax.jit(
        random_exploration_policy, backend='cpu')

  def select_action(self, observation: types.NestedArray):
    if self._num_observations > self._min_observations:
      return self._actor.select_action(observation)
    else:
      action, self._key = self._random_exploration_policy(
          self._key, observation)
      return utils.to_numpy(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    return self._actor.observe_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    self._num_observations += 1
    self._actor.observe(action, next_timestep)

  def update(self, wait: bool = True):
    if self._num_observations < self._min_observations:
      return
    self._learner.step()
    self._actor.update(wait=wait)

  def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
    return self._learner.get_variables(names)
