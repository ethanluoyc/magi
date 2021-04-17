from collections import deque

import numpy as np
from gym.spaces import Box, Discrete

class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(
        self,
        buffer_size,
        state_space,
        action_space,
        gamma,
        nstep,
    ):
        assert len(state_space.shape) in (1, 3)

        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.nstep = nstep
        self.state_shape = state_space.shape
        self.use_image = len(self.state_shape) == 3

        if self.use_image:
            # Store images as a list of LazyFrames, which uses 4 times less memory.
            self.state = [None] * buffer_size
            self.next_state = [None] * buffer_size
        else:
            self.state = np.empty((buffer_size, *state_space.shape), dtype=np.float32)
            self.next_state = np.empty((buffer_size, *state_space.shape), dtype=np.float32)

        if type(action_space) == Box:
            self.action = np.empty((buffer_size, *action_space.shape), dtype=np.float32)
        elif type(action_space) == Discrete:
            self.action = np.empty((buffer_size, 1), dtype=np.int32)
        else:
            NotImplementedError

        self.reward = np.empty((buffer_size), dtype=np.float32)
        self.done = np.empty((buffer_size), dtype=np.float32)

    def append(self, state, action, reward, done, next_state):
        self._append(state, action, reward, done, next_state)

    def _append(self, state, action, reward, done, next_state):
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = float(reward)
        self.done[self._p] = float(done)
        self.next_state[self._p] = next_state

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def _sample_idx(self, batch_size):
        return np.random.randint(low=0, high=self._n, size=batch_size)

    def _sample(self, idxes):
        if self.use_image:
            state = np.empty((len(idxes), *self.state_shape), dtype=np.uint8)
            next_state = state.copy()
            for i, idx in enumerate(idxes):
                state[i, ...] = self.state[idx]
                next_state[i, ...] = self.next_state[idx]
        else:
            state = self.state[idxes]
            next_state = self.next_state[idxes]

        return (
            state,
            self.action[idxes],
            self.reward[idxes],
            self.done[idxes],
            next_state,
        )

    def sample(self, batch_size):
        idxes = self._sample_idx(batch_size)
        batch = self._sample(idxes)
        # Use fake weight to use the same interface with PER.
        weight = np.ones((), dtype=np.float32)
        return weight, batch
