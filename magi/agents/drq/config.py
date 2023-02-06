"""Configuration for DrQ agent."""
import dataclasses
from typing import Optional

from acme.adders import reverb as adders_reverb

from magi.agents.drq import augmentations


@dataclasses.dataclass
class DrQConfig:
    """Configuration parameters for DrQ."""

    target_entropy: float
    augmentation: augmentations.DataAugmentation = augmentations.batched_random_crop

    min_replay_size: int = 1_000
    max_replay_size: int = 1_000_000
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
    prefetch_size: Optional[int] = None

    discount: float = 0.99
    batch_size: int = 128
    initial_num_steps: int = 1000

    critic_learning_rate: float = 3e-4
    critic_target_update_frequency: int = 1
    critic_q_soft_update_rate: float = 0.005

    actor_learning_rate: float = 3e-4
    actor_update_frequency: int = 1

    temperature_learning_rate: float = 3e-4
    temperature_adam_b1: float = 0.5
    init_temperature: float = 0.1
