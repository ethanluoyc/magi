"""Configuration for MPO agent"""
import dataclasses
from typing import Optional

from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class MPOConfig:
  """MPO agent parameters."""

  min_replay_size: int = 1000
  max_replay_size: int = 1_000_000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  prefetch_size: Optional[int] = 4
  samples_per_insert: float = 32.0

  discount: float = 0.99
  batch_size: int = 256
  num_samples: int = 20
  n_step: int = 5
  clipping: bool = True
  policy_learning_rate = 1e-4
  critic_learning_rate = 1e-4
  dual_learning_rate = 1e-2
  target_policy_update_period: int = 100
  target_critic_update_period: int = 100
  retrace: bool = True
  retrace_sequence_length: int = 10
