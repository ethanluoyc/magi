import dataclasses
from typing import Optional

from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class SACConfig:
    min_replay_size: int = 1
    max_replay_size: int = 1_000_000
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
    prefetch_size: Optional[int] = None

    discount: float = 0.99
    batch_size: int = 256
    initial_num_steps: int = 10000

    critic_learning_rate: float = 1e-3
    critic_target_update_frequency: int = 2
    critic_soft_update_rate: float = 0.01

    actor_learning_rate: float = 1e-3
    actor_update_frequency: int = 2

    max_gradient_norm: float = 0.5

    temperature_learning_rate: float = 1e-4
    temperature_adam_b1: float = 0.5
    init_temperature: float = 0.1
