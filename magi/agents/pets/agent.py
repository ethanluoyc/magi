import dm_env
import numpy as np
from acme import core


class ModelBasedAgent(core.Actor):

  def __init__(self, actor: core.Actor, learner: core.Learner):
    self._actor = actor
    self._learner = learner

    self._last_timestep = None

  def select_action(self, observation: np.ndarray):
    return self._actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)
    self._last_timestep = timestep

  def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep):
    self._actor.observe(action, next_timestep)
    self._last_timestep = next_timestep

  def update(self):
    # Perform update at the end of an episode.
    if self._last_timestep.last():
      self._learner.step()
      self._actor.update()
