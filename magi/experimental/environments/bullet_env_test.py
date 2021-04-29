from absl.testing import absltest
import numpy as np
import numpy.testing as np_test

from magi.experimental.environments import bullet_cartpole_env
from magi.experimental.environments import bullet_kuka_env


class BulletEnvTest(absltest.TestCase):

  def test_env(self):
    env = bullet_cartpole_env.CartPoleBulletEnv()
    env2 = bullet_cartpole_env.CartPoleBulletEnv()

    def get_state(env):
      p = env._p
      return np.array(
          p.getJointState(env.cartpole, 1)[0:2] + p.getJointState(env.cartpole, 0)[0:2])

    env.reset()
    original_state = get_state(env)
    env2.reset()
    original_state2 = get_state(env)
    np_test.assert_allclose(original_state, original_state2)

  def test_env_same_obs_with_same_seed(self):
    env = bullet_cartpole_env.CartPoleBulletEnv()
    env2 = bullet_cartpole_env.CartPoleBulletEnv()

    def get_state(env):
      p = env._p
      return np.array(
          p.getJointState(env.cartpole, 1)[0:2] + p.getJointState(env.cartpole, 0)[0:2])

    env.seed(0)
    env.reset()
    original_state = get_state(env)
    env2.seed(0)
    env2.reset()
    original_state2 = get_state(env2)
    np_test.assert_allclose(original_state, original_state2)

  def test_env_different_obs_with_different_seed(self):
    env = bullet_cartpole_env.CartPoleBulletEnv()
    env2 = bullet_cartpole_env.CartPoleBulletEnv()

    def get_state(env):
      p = env._p
      return np.array(
          p.getJointState(env.cartpole, 1)[0:2] + p.getJointState(env.cartpole, 0)[0:2])

    env.seed(0)
    env.reset()
    original_state = get_state(env)
    env2.seed(1)
    env2.reset()
    original_state2 = get_state(env2)
    # np_test.assert_allclose(original_state, original_state2)
    self.assertFalse(np.allclose(original_state, original_state2))


class BulletKukaEnv(absltest.TestCase):

  @absltest.skip("Known failure with egl on")
  def test_env(self):
    env = bullet_kuka_env.KukaDiverseObjectEnv()
    env2 = bullet_kuka_env.KukaDiverseObjectEnv()

    def get_state(env):
      return env._get_observation()

    env.reset()
    original_state = get_state(env)
    env2.reset()
    original_state2 = get_state(env)
    np_test.assert_allclose(original_state, original_state2)

  # def test_env_same_obs_with_same_seed(self):
  #   env = bullet_cartpole_env.CartPoleBulletEnv()
  #   env2 = bullet_cartpole_env.CartPoleBulletEnv()
  #   def get_state(env):
  #     p = env._p
  #     return np.array(p.getJointState(env.cartpole, 1)[0:2]
  #                     + p.getJointState(env.cartpole, 0)[0:2])
  #   env.seed(0)
  #   env.reset()
  #   original_state = get_state(env)
  #   env2.seed(0)
  #   env2.reset()
  #   original_state2 = get_state(env2)
  #   np_test.assert_allclose(original_state, original_state2)

  # def test_env_different_obs_with_different_seed(self):
  #   env = bullet_cartpole_env.CartPoleBulletEnv()
  #   env2 = bullet_cartpole_env.CartPoleBulletEnv()
  #   def get_state(env):
  #     p = env._p
  #     return np.array(p.getJointState(env.cartpole, 1)[0:2]
  #                     + p.getJointState(env.cartpole, 0)[0:2])
  #   env.seed(0)
  #   env.reset()
  #   original_state = get_state(env)
  #   env2.seed(1)
  #   env2.reset()
  #   original_state2 = get_state(env2)
  #   # np_test.assert_allclose(original_state, original_state2)
  #   self.assertFalse(np.allclose(original_state, original_state2))


if __name__ == '__main__':
  absltest.main()
