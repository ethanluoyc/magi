from typing import Sized

import jax.numpy as jnp
import numpy as np

# TODO: make this acme.adders compatible.


class Transition:

  def __init__(self, o_tm1, a_t, r_t, o_t):
    self.o_tm1 = np.asarray(o_tm1)
    self.a_t = np.asarray(a_t)
    self.r_t = np.asarray(r_t)
    self.o_t = np.asarray(o_t)

  def __getitem__(self, ind) -> 'Transition':
    return Transition(self.o_tm1[ind], self.a_t[ind], self.r_t[ind], self.o_t[ind])

  def __len__(self):
    return len(self.o_tm1)


class TransitionIterator:
  """Iterator for batches of transitions. Adapted from
    https://github.com/facebookresearch/mbrl-lib/blob/master/mbrl/util/replay_buffer.py
  """

  def __init__(self,
               transition: Transition,
               batch_size: int,
               shuffle_each_epoch: bool = True,
               rng=None):
    self._transition = transition
    self.batch_size = batch_size
    self.num_stored = len(transition)
    self._order: np.ndarray = np.arange(self.num_stored)
    self._current_batch = 0
    self._shuffle_each_epoch = shuffle_each_epoch
    self._rng = rng if rng is not None else np.random.default_rng()

  def _get_indices_next_batch(self) -> Sized:
    start_idx = self._current_batch * self.batch_size
    if start_idx >= self.num_stored:
      raise StopIteration
    end_idx = min((self._current_batch + 1) * self.batch_size, self.num_stored)
    order_indices = range(start_idx, end_idx)
    indices = self._order[order_indices]
    self._current_batch += 1
    return indices

  def __iter__(self):
    self._current_batch = 0
    if self._shuffle_each_epoch:
      self._order = self._rng.permutation(self.num_stored)
    return self

  def __next__(self):
    return self._transition[self._get_indices_next_batch()]


class Dataset:

  def __init__(self):
    self.observations_t = []
    self.observations_tp1 = []
    self.rewards = []
    self.actions = []

  def add(self, obs_t, obs_tp1, action, reward):
    self.observations_t.append(obs_t)
    self.observations_tp1.append(obs_tp1)
    self.rewards.append(reward)
    self.actions.append(action)

  def _collect(self):
    return (
        jnp.stack(self.observations_t),
        jnp.stack(self.observations_tp1),
        jnp.stack(self.actions),
        jnp.stack(self.rewards),
    )

  def get_iterators(self,
                    batch_size: int,
                    val_ratio: float,
                    shuffle: bool = True,
                    rng=None):
    o_tm1, o_t, a_t, r_t = self._collect()
    transition = Transition(o_tm1, a_t, r_t, o_t)
    if rng is None:
      rng = np.random.default_rng()
    dataset_size = len(transition)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    ind = np.arange(dataset_size)
    if shuffle:
      rng.shuffle(ind)
    if val_size == 0:
      train_iterator = TransitionIterator(transition[ind],
                                          batch_size=batch_size,
                                          rng=rng)
      return train_iterator, None
    else:
      train_ind = ind[:train_size]
      val_ind = ind[train_size:]
      train_iterator = TransitionIterator(transition[train_ind],
                                          batch_size=batch_size,
                                          rng=rng)
      val_iterator = TransitionIterator(transition[val_ind],
                                        batch_size=batch_size,
                                        rng=rng)
      return train_iterator, val_iterator

  def reset(self):
    self.observations_t = []
    self.observations_tp1 = []
    self.rewards = []
    self.actions = []
