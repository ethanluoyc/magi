import dataclasses
from typing import Optional

from acme.adders import reverb as adders_reverb
import optax


@dataclasses.dataclass
class TD3BCConfig:
    policy_optimizer: Optional[optax.GradientTransformation] = None
    critic_optimizer: Optional[optax.GradientTransformation] = None
    discount: float = 0.99

    batch_size: int = 256
    # target network soft update rate
    tau: float = 0.005
    # coefficient controlling relative importance of BC and TD loss
    alpha: float = 2.5
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    policy_update_period: int = 2

    # Replay options
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
    # samples_per_insert: float = 256 * 4
    # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
    # See a formula in make_replay_tables for more details.
    # samples_per_insert_tolerance_rate: float = 0.1
    min_replay_size: int = 1
    max_replay_size: int = 1000000
    prefetch_size: Optional[int] = None
