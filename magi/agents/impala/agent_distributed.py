"""Defines the distributed IMPALA agent class."""

from typing import Any, Callable, Dict, Optional

from acme import specs
import dm_env
import haiku as hk

from magi.agents.impala import builder as builder_lib
from magi.agents.impala import config as config_lib
from magi.agents.impala import distributed_layout as layout


class DistributedIMPALA(layout.IMPALADistributedLayout):
  """Program definition for IMPALA."""

  def __init__(
      self,
      environment_factory: Callable[[int, bool], dm_env.Environment],
      network_factory: Callable[[specs.BoundedArray], Dict[str, Any]],
      num_actors: int = 1,
      num_caches: int = 0,
      environment_spec: specs.EnvironmentSpec = None,
      config: Optional[config_lib.IMPALAConfig] = None,
      max_actor_steps: Optional[int] = None,
      log_every: float = 5.0,
      seed: int = 0,
      logger_fn=None,
  ):
    if not environment_spec:
      environment_spec = specs.make_environment_spec(
          environment_factory(seed, False))
    network_spec = network_factory(environment_spec.actions)
    initial_state = hk.without_apply_rng(
        hk.transform(network_spec['initial_state'])).apply(None)
    builder = builder_lib.IMPALABuilder(config, initial_state)
    super().__init__(
        environment_factory=environment_factory,
        network_factory=network_factory,
        builder=builder,
        num_actors=num_actors,
        num_caches=num_caches,
        environment_spec=environment_spec,
        max_actor_steps=max_actor_steps,
        log_every=log_every,
        seed=seed,
        logger_fn=logger_fn,
    )
