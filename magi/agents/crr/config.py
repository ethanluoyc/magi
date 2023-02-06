"""Configuration parameters for CRR."""
import dataclasses
from typing import Optional

import optax
from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class CRRConfig:
    """Configuration parameters for CRR."""

    policy_optimizer: Optional[optax.GradientTransformation] = None
    critic_optimizer: Optional[optax.GradientTransformation] = None
    discount: float = 0.99
    target_update_period: int = 100
    num_action_samples_td_learning: int = 1
    num_action_samples_policy_weight: int = 4
    baseline_reduce_function: str = "mean"
    policy_improvement_modes: str = "exp"
    ratio_upper_bound: float = 20.0
    beta: float = 1.0
    batch_size: int = 256

    # Replay options
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
    # samples_per_insert: float = 256 * 4
    # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
    # See a formula in make_replay_tables for more details.
    # samples_per_insert_tolerance_rate: float = 0.1
    min_replay_size: int = 1
    max_replay_size: int = 1000000
    prefetch_size: Optional[int] = None
