"""Test for subprocess wrapper."""
from absl.testing import absltest
import gym

from magi.wrappers import gym_subproc


class SubProcessWrapperTest(absltest.TestCase):
    def test_subproc_env(self):
        """Test that subproc env runs"""
        env = gym_subproc.SubprocEnv(lambda **kwargs: gym.make("CartPole-v1"))
        env.seed()
        env.reset()
        while True:
            _, _, done, _ = env.step(env.action_space.sample())
            if done:
                break


if __name__ == "__main__":
    absltest.main()
