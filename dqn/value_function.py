import torch
from torch.autograd import Variable


class Q(object):
    def __init__(self, dqn):
        self.dqn = dqn

    def argmax(self, state):
        values = self.values(state)
        _, action = values.max(dim=1, keepdim=True)
        return action.data.cpu().numpy()

    def values(self, state):
        state = Variable(torch.from_numpy(state), requires_grad=False)
        return self.dqn(state)

    def value(self, state, action):
        action = Variable(torch.from_numpy(action), requires_grad=False)
        return self.values(state).gather(dim=1, index=action)

    def target_value(self, state, reward, gamma=0.99):
        reward = Variable(torch.from_numpy(reward), requires_grad=False)
        values, _ = self.values(state).max(dim=1, keepdim=True)
        return reward + gamma * values
