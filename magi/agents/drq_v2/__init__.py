"""DrQ-v2 agent implementation."""
from magi.agents.drq_v2.acting import DrQV2Actor  # noqa
from magi.agents.drq_v2.agent import DrQV2  # noqa
from magi.agents.drq_v2.config import DrQV2Config  # noqa
from magi.agents.drq_v2.learning import DrQV2Learner  # noqa
from magi.agents.drq_v2.networks import DrQV2Networks  # noqa
from magi.agents.drq_v2.networks import get_default_behavior_policy  # noqa
from magi.agents.drq_v2.networks import make_networks  # noqa
