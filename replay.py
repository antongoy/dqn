import numpy as np


class StateGenerator(object):
    def __init__(self, capacity, observation_shape):
        self.capacity = capacity
        self.observation_shape = observation_shape

        self.observations = np.zeros((capacity,) + self.observation_shape, dtype=np.float32)
        self.counter = 0

    def index(self, counter):
        return counter % self.capacity

    def register_observation(self, observation):
        self.counter += 1
        index = self.index(self.counter)
        self.observations[index] = observation

    def produce_state(self):
        observations_idx = [self.index(self.counter - idx) for idx in range(self.capacity)]
        return self.observations[observations_idx]


class ExperienceReplay(object):
    def __init__(self, capacity, state_size, observation_shape):
        self.capacity = capacity
        self.state_size = state_size
        self.observation_shape = observation_shape

        self.observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)

        self.counter = -1

    def index(self, counter):
        return counter % self.capacity

    def register_initial_observation(self, observation):
        self.counter += 1

        self.observations[self.index(self.counter)] = observation
        for _ in range(self.state_size - 1):
            self.observations[self.index(self.counter - 1)] = np.zeros(self.observation_shape, dtype=np.float32)

        return self.produce_state()

    def register_transition(self, new_observation, action, reward):
        self.counter += 1
        index = self.index(self.counter)
        prev_index = self.index(self.counter - 1)

        self.observations[index] = new_observation
        self.actions[prev_index] = action
        self.rewards[prev_index] = reward

        return self.produce_state()

    def reset(self):
        self.counter = 0

    def state(self, state_idx):
        observations_idx = [self.index(state_idx - idx) for idx in range(self.state_size)]
        return self.observations[observations_idx]

    def produce_state(self):
        return self.state(self.counter)

    def sample_minibatch(self, batch_size):
        indexes = np.random.choice(min(self.capacity, self.counter), size=min(batch_size, self.counter), replace=False)

        states = np.stack([self.state(idx + 1) for idx in indexes], axis=0)
        prev_states = np.stack([self.state(idx) for idx in indexes], axis=0)

        return prev_states, self.actions[indexes], self.rewards[indexes], states
