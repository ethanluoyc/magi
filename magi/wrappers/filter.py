"""Environment wrappers for filtering observations."""
from acme.wrappers import base


class TakeKeyWrapper(base.EnvironmentWrapper):
    """Wraps a control environment and adds a rendered pixel observation."""

    def __init__(self, environment, key):
        super().__init__(environment)
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
