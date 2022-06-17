"""Soft Actor-Critic agent parameters."""
import dataclasses
from typing import Optional

from acme import specs
from acme.adders import reverb as adders_reverb
import numpy as np


def target_entropy_from_env_spec(env_spec: specs.EnvironmentSpec) -> float:
  """Compute the heuristic target entropy"""
  return -float(np.prod(env_spec.actions.shape))


@dataclasses.dataclass
class SACConfig:
  """Soft Actor-Critic agent parameters."""

  entropy_coefficient: Optional[float] = None
  target_entropy: float = 0
  min_replay_size: int = 1000
  max_replay_size: int = 1_000_000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  prefetch_size: Optional[int] = None

  discount: float = 0.99
  batch_size: int = 256

  critic_learning_rate: float = 1e-3
  critic_soft_update_rate: float = 0.005

  actor_learning_rate: float = 1e-3

  temperature_learning_rate: float = 1e-3
  temperature_adam_b1: float = 0.5
  init_temperature: float = 0.1
