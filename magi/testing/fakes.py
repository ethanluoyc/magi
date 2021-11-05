"""Fake objects used for testing."""
from typing import Sequence

from acme import specs
from acme.testing import fakes
import numpy as np


class ContinuousVisualEnvironment(fakes.Environment):
    """Continuous state and action fake environment."""

    def __init__(
        self,
        *,
        action_dim: int = 1,
        observation_shape: Sequence[int] = (32, 32, 3),
        bounded: bool = False,
        dtype=np.float32,
        reward_dtype=np.float32,
        **kwargs,
    ):
        """Initialize the environment.

        Args:
          action_dim: number of action dimensions.
          observation_dim: number of observation dimensions.
          bounded: whether or not the actions are bounded in [-1, 1].
          dtype: dtype of the action and observation spaces.
          reward_dtype: dtype of the reward and discounts.
          **kwargs: additional kwargs passed to the Environment base class.
        """

        action_shape = () if action_dim == 0 else (action_dim,)

        observations = specs.Array(observation_shape, dtype)
        rewards = specs.Array((), reward_dtype)
        discounts = specs.BoundedArray((), reward_dtype, 0.0, 1.0)

        if bounded:
            actions = specs.BoundedArray(action_shape, dtype, -1.0, 1.0)
        else:
            actions = specs.Array(action_shape, dtype)

        super().__init__(
            spec=specs.EnvironmentSpec(
                observations=observations,
                actions=actions,
                rewards=rewards,
                discounts=discounts,
            ),
            **kwargs,
        )
