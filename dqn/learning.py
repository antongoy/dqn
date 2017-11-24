import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy


class NoLearning(object):
    def learn(self, states, actions, rewards, next_states):
        pass


class QLearning(object):
    def __init__(self, env, dqn, update_every=10000, learn_every=5, lr=0.0001, gamma=0.99):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.learn_every = learn_every
        self.update_every = update_every

        self.dqn = dqn
        self.dqn_ = deepcopy(dqn)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn.model.parameters(), lr=lr)

    def learn(self, states, actions, rewards, next_states):
        if self.env.frame and self.env.frame % self.learn_every:
            return

        values = self.dqn.value(states, actions)
        target_values = self.dqn_.target_value(next_states, rewards, gamma=self.gamma)

        loss = self.criterion(values, target_values.detach())
        # loss = torch.clamp(loss, -1, 1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.env.frame and not self.env.frame % self.update_every:
            self.dqn_.model.load_state_dict(self.dqn.model.state_dict())
