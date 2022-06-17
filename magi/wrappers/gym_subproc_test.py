"""Test for subprocess wrapper."""
from absl.testing import absltest
import gym
from gym import spaces

from magi.wrappers import gym_subproc


class DummyEnv(gym.Env):
  """Dummy gym environment for testing subprocess wrapper"""

  def __init__(self):
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Discrete(2)

  def step(self, action):
    del action
    return self.observation_space.sample(), 0.0, True, {}

  def reset(self):
    return self.observation_space.sample()

  def render(self, mode='human'):
    del mode


class SubProcessWrapperTest(absltest.TestCase):

  def test_subproc_env(self):
    """Test that subproc env runs"""
    env = gym_subproc.SubprocEnv(lambda **_: DummyEnv())
    env.seed()
    env.reset()
    while True:
      _, _, done, _ = env.step(env.action_space.sample())
      if done:
        break


if __name__ == '__main__':
  absltest.main()
