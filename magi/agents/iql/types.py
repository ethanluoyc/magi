"""Type definition for IQL agent."""
import collections
from typing import Any, Dict, Sequence, Tuple

import flax
import numpy as np

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, np.ndarray]

TimeStep = Tuple[np.ndarray, float, bool, dict]
