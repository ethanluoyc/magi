import collections

from acme import types
from acme.wrappers import base
import dm_env
from dm_env import specs as dm_env_specs
import numpy as np
import tree

class ConcatFrameWrapper(base.EnvironmentWrapper):
  """Wrapper that stacks observations along a new final axis."""

  def __init__(self, environment: dm_env.Environment):
    """Initializes a new FrameStackingWrapper.

    Args:
      environment: Environment.
      num_frames: Number frames to stack.
    """
    self._environment = environment
    original_spec = self._environment.observation_spec()
    self._observation_spec = tree.map_structure(
        lambda spec: self._update_spec(spec), original_spec)

  def _process_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    observation = tree.map_structure(lambda x: x.reshape(x.shape[:-2] + (-1,)),
                                     timestep.observation)
    return timestep._replace(observation=observation)

  def reset(self) -> dm_env.TimeStep:
    return self._process_timestep(self._environment.reset())

  def step(self, action: int) -> dm_env.TimeStep:
    return self._process_timestep(self._environment.step(action))

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec

  def _update_spec(self, spec):
    return dm_env_specs.Array(shape=spec.shape[:-2] + (np.prod(spec.shape[-2:]),),
                              dtype=spec.dtype,
                              name=spec.name)