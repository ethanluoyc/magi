# python3
"""Importance weighted advantage actor-critic (IMPALA) agent implementation."""

from typing import Callable

from acme import specs
from acme.jax import networks
from acme.utils import counting
from acme.utils import loggers
import haiku as hk

from magi.agents.impala import builder as builder_lib
from magi.agents.impala import config as config_lib
from magi.agents.impala import local_layout as layout


class IMPALAFromConfig(layout.IMPALALocalLayout):
  """IMPALA Agent."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      forward_fn: networks.PolicyValueRNN,
      unroll_fn: networks.PolicyValueRNN,
      initial_state_fn: Callable[[], hk.LSTMState],
      config: config_lib.IMPALAConfig,
      seed: int = 0,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
  ):
    network_dict = {
        'initial_state': initial_state_fn,
        'forward': forward_fn,
        'unroll': unroll_fn,
    }
    initial_state = hk.without_apply_rng(
        hk.transform(initial_state_fn)).apply(None)
    builder = builder_lib.IMPALABuilder(
        config,
        initial_state=initial_state,
    )
    super().__init__(
        network_dict,
        environment_spec,
        builder,
        batch_size=config.batch_size,
        seed=seed,
        counter=counter,
        logger=logger,
    )
