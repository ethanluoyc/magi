"""TD3 agent configuration."""
import dataclasses
from typing import Optional

from acme.adders import reverb as adders_reverb
import optax


@dataclasses.dataclass
class TD3Config:
    """Configuration for TD3 agent."""

    # Minimum size for the replay buffer.
    # This also determines the number of environment steps to take
    # before calling `learner.step`
    min_replay_size: int = 1_000
    # Capacity of the replay buffer.
    max_replay_size: int = 1_000_000
    # Reverb replay table name.
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

    # Prefetch size from Reverb replay
    prefetch_size: Optional[int] = None
    # Batch size for sampling transitions
    batch_size: int = 256

    # Optimizer for the policy
    policy_optimizer: Optional[optax.GradientTransformation] = None
    # Optimizer for the critic
    critic_optimizer: Optional[optax.GradientTransformation] = None

    # Additional discounts (usually referred to as gamma in the literature)
    discount: float = 0.99
    # Rate for updating the target parameters by Polyak averaging (tau)
    soft_update_rate: float = 0.005
    # Noise added to the actions used by the critic
    policy_noise: float = 0.2
    # Clipping parameter for the noise
    policy_noise_clip: float = 0.5
    # Exploration noise for the behavior policy
    policy_exploration_noise: float = 0.1
    # Number of actor steps for every policy and target network update
    policy_update_period: int = 2
