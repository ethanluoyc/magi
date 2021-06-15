from absl.testing import absltest
import numpy as np
from magi.agents.pets import dataset


class DataIteratorTest(absltest.TestCase):
  """Tests for dataset iterators"""
  def test_transition_iterator(self):
    obs_size = 3
    act_size = 4
    ds = dataset.ReplayBuffer(int(1e3), obs_shape=(obs_size,), action_shape=(act_size,))
    for _ in range(10):
      ds.add(np.zeros(obs_size, ), np.zeros(act_size), np.zeros(obs_size, ), 1.0, False)
    train_iterator, val_iterator = ds.get_iterators(batch_size=10, val_ratio=0.1)
    for batch in train_iterator:
      self.assertEqual(batch.obs.shape[-1], obs_size)
    for batch in val_iterator:
      self.assertEqual(batch.obs.shape[-1], obs_size)


if __name__ == "__main__":
  absltest.main()
