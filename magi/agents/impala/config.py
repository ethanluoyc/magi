"""Parameters for IMPALA agent."""
import dataclasses

from acme.adders import reverb as reverb_adders
import numpy as np


@dataclasses.dataclass
class IMPALAConfig:
  """IMPALA agent configuration."""

  replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
  sequence_length: int = 100
  sequence_period: int = 100
  discount: float = 0.99
  max_queue_size: int = 16 * 10
  batch_size: int = 16
  learning_rate: float = 1e-3
  entropy_cost: float = 0.01
  baseline_cost: float = 0.5
  max_abs_reward: float = np.inf
  max_gradient_norm: float = np.inf
  break_end_of_episode: bool = False
