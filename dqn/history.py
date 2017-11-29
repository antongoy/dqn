import torch
import numpy as np


class RingBuffer(object):
    def __init__(self, entity_shape, max_capacity, dtype):
        self.entity_shape = entity_shape
        self.max_capacity = max_capacity
        self.dtype = dtype
        self.counter = 0
        self.buffer = np.zeros((max_capacity,) + entity_shape, dtype=dtype)

    def append(self, entity=None):
        if entity is not None:
            self.buffer[self.counter % self.max_capacity] = entity

        self.counter += 1

    def __getitem__(self, item):
        return self.buffer[item % self.max_capacity]

    def last(self, n):
        return self.buffer[None, [(self.counter - idx) % self.max_capacity for idx in range(1, n + 1)]]

    def reset(self):
        self.counter = 0
        self.buffer[:] = 0


class StateGenerator(object):
    def __init__(self, observation_shape, state_size):
        self.observation_shape = observation_shape
        self.state_size = state_size
        self.observations = RingBuffer(observation_shape, state_size, dtype=np.uint8)

    def produce_state(self, observation):
        self.observations.append(observation)
        return self.observations.last(self.state_size)

    def reset(self):
        self.observations.reset()


class History(object):
    def __init__(self, observation_shape, state_size, max_capacity):
        self.max_capacity = max_capacity
        self.state_size = state_size
        self.observation_shape = observation_shape

        self.states = RingBuffer((state_size,) + observation_shape, max_capacity, dtype=np.uint8)
        self.actions = RingBuffer((), max_capacity, dtype=np.int32)
        self.rewards = RingBuffer((), max_capacity, dtype=np.int32)
        self.isinit = RingBuffer((), max_capacity, dtype=np.int32)

        self.counter = 0

    def register_transition(self, action, reward, state, isterminal=False):
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(state)
        self.isinit.append(isterminal)
        self.counter += 1

    def register_init_state(self, state):
        self.actions.append()
        self.rewards.append()
        self.states.append(state)
        self.isinit.append(True)
        self.counter += 1

    def batch(self, batch_size):
        indexes = np.random.choice(min(self.max_capacity, self.counter), size=batch_size, replace=False)
        indexes = [idx for idx in indexes if not self.isinit[idx]]

        prev_states = np.stack([self.states[idx - 1] for idx in indexes])
        actions = np.vstack([self.actions[idx] for idx in indexes])
        rewards = np.vstack([self.rewards[idx] for idx in indexes])
        states = np.stack([self.states[idx] for idx in indexes])
        isterminals = np.vstack([self.isinit[idx + 1] for idx in indexes])

        return prev_states, actions, rewards, states, isterminals
