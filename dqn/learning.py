import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy

from .value_function import Q


class NoLearning(object):
    def learn(self, states, actions, rewards, next_states):
        pass


class QLearning(object):
    def __init__(self, env, dqn, update_every=10000, learn_every=5, lr=0.0001, gamma=0.99, batch_size=32):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.update_every = update_every
        self.target_dqn = dqn
        self.learning_dqn = deepcopy(dqn)

        self.Q = Q(self.learning_dqn)
        self.Q_ = Q(dqn)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.learning_dqn.parameters(), lr=lr)

    def learn(self, states, actions, rewards, next_states):
        if self.env.frame and self.env.frame % self.learn_every:
            return

        values = self.Q.value(states, actions)
        target_values = self.Q_.target_value(next_states, rewards, gamma=self.gamma)

        loss = self.criterion(values, target_values.detach())
        loss = torch.clamp(loss, -1, 1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.env.frame and not self.env.frame % self.update_every:
            self.target_dqn.load_state_dict(self.learning_dqn.state_dict())
