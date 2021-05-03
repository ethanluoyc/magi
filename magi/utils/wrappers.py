import dm_env

from acme import types
from acme.wrappers import base
from dm_env import specs as dm_env_specs
import numpy as np
import tree

import collections


class FrameStackingWrapper(base.EnvironmentWrapper):
  """Wrapper that stacks observations along a new final axis."""

  def __init__(self, environment: dm_env.Environment, num_frames: int = 4):
    """Initializes a new FrameStackingWrapper.

    Args:
      environment: Environment.
      num_frames: Number frames to stack.
    """
    self._environment = environment
    original_spec = self._environment.observation_spec()
    self._stackers = tree.map_structure(lambda _: FrameStacker(num_frames=num_frames),
                                        self._environment.observation_spec())
    self._observation_spec = tree.map_structure(
        lambda stacker, spec: stacker.update_spec(spec), self._stackers, original_spec)

  def _process_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    observation = tree.map_structure(lambda stacker, x: stacker.step(x), self._stackers,
                                     timestep.observation)
    return timestep._replace(observation=observation)

  def reset(self) -> dm_env.TimeStep:
    for stacker in tree.flatten(self._stackers):
      stacker.reset()
    return self._process_timestep(self._environment.reset())

  def step(self, action: int) -> dm_env.TimeStep:
    return self._process_timestep(self._environment.step(action))

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec


class FrameStacker:
  """Simple class for frame-stacking observations."""

  def __init__(self, num_frames: int):
    self._num_frames = num_frames
    self.reset()

  @property
  def num_frames(self) -> int:
    return self._num_frames

  def reset(self):
    self._stack = collections.deque(maxlen=self._num_frames)

  def step(self, frame: np.ndarray) -> np.ndarray:
    if not self._stack:
      # Fill stack with blank frames if empty.
      self._stack.extend([frame] * (self._num_frames - 1))
    self._stack.append(frame)
    return np.concatenate(self._stack, axis=-1)

  def update_spec(self, spec: dm_env_specs.Array) -> dm_env_specs.Array:
    return dm_env_specs.Array(shape=spec.shape[:-1] +
                              (spec.shape[-1] * self._num_frames,),
                              dtype=spec.dtype,
                              name=spec.name)


class TakeKeyWrapper(base.EnvironmentWrapper):
  """Wraps a control environment and adds a rendered pixel observation."""

  def __init__(self, environment, key='pixels'):
    self._environment = environment
    self._key = key

  def reset(self):
    time_step = self._environment.reset()
    return time_step._replace(observation=time_step.observation[self._key])

  def step(self, action):
    time_step = self._environment.step(action)
    return time_step._replace(observation=time_step.observation[self._key])

  def observation_spec(self):
    return self._environment.observation_spec()[self._key]

  def action_spec(self):
    return self._environment.action_spec()

  def __getattr__(self, name):
    return getattr(self._environment, name)
