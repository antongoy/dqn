import torch
import numpy as np


class History(object):
    def __init__(self, capacity, state_size, observation_shape, batch_size=32):
        self.capacity = capacity
        self.state_size = state_size
        self.batch_size = batch_size
        self.observation_shape = observation_shape

        self.observations = np.zeros((capacity,) + observation_shape, dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int8)
        self.rewards = np.zeros(capacity, dtype=np.int8)

        self.counter = -1

    def _index(self, counter):
        return counter % self.capacity

    def _state(self, state_idx):
        observations_idx = [self._index(state_idx - idx) for idx in range(self.state_size)]
        return self.observations[np.newaxis, observations_idx].astype(np.float32)

    def _register_observation(self, observation):
        self.counter += 1
        self.observations[self._index(self.counter)] = observation

    def _register_zero_observations(self, n):
        for _ in range(n):
            self._register_observation(np.zeros(self.observation_shape, dtype=np.float32))

    def register_init_observation(self, observation):
        self._register_zero_observations(self.state_size - 1)
        self._register_observation(observation)
        return self._state(self.counter)

    def register_transition(self, new_observation, action, reward):
        self.counter += 1
        index = self._index(self.counter)
        prev_index = self._index(self.counter - 1)

        self.observations[index] = new_observation
        self.actions[prev_index] = action
        self.rewards[prev_index] = reward

        return self._state(self.counter)

    def reset(self):
        self.counter = 0

    def batch(self):
        indexes = np.random.choice(min(self.capacity, self.counter), size=min(self.batch_size, self.counter), replace=False)

        states = np.vstack([self._state(idx + 1) for idx in indexes])
        actions = self.actions[indexes, np.newaxis].astype(np.int64)
        rewards = self.rewards[indexes, np.newaxis].astype(np.float32)
        prev_states = np.vstack([self._state(idx) for idx in indexes])

        return prev_states, actions, rewards, states
