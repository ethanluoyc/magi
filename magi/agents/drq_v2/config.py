"""Configuration for DrQV2."""
import dataclasses
from typing import Tuple

from acme.adders import reverb as adders_reverb

from magi.agents.drq import augmentations


@dataclasses.dataclass
class DrQV2Config:
  """Configuration parameters for DrQ."""

  augmentation: augmentations.DataAugmentation = augmentations.batched_random_crop

  min_replay_size: int = 2_000
  max_replay_size: int = 1_000_000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  prefetch_size: int = 1

  discount: float = 0.99
  batch_size: int = 256
  n_step: int = 3

  critic_q_soft_update_rate: float = 0.01
  learning_rate: float = 1e-4
  noise_clip: float = 0.3
  sigma: Tuple[float, float, int] = (1.0, 0.1, 500000)

  samples_per_insert: float = 256.0
  # Rate to be used for the SampleToInsertRatio rate limiter tolerance.
  # See a formula in make_replay_tables for more details.
  # NOTE(yl) this is currently unused and disabled by the single-process agent.
  # We should enable this when we implement the distribtued version.
  samples_per_insert_tolerance_rate: float = 0.1
