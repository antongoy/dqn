import torch
import torch.nn as nn

from torch.autograd import Variable


class WrapperDQN(object):
    def __init__(self, dqn, cuda=False):
        self.model = dqn
        self.cuda = cuda

        if self.cuda:
            self.model.cuda()

    def argmax(self, state):
        values = self.values(state)
        _, action = values.max(dim=1, keepdim=True)
        return action.data.cpu().numpy()

    def values(self, state):
        state = Variable(torch.from_numpy(state), requires_grad=False)
        if self.cuda:
            state.cuda()
        return self.model(state)

    def value(self, state, action):
        action = Variable(torch.from_numpy(action), requires_grad=False)
        if self.cuda:
            action.cuda()
        return self.values(state).gather(dim=1, index=action)

    def target_value(self, state, reward, gamma=0.99):
        reward = Variable(torch.from_numpy(reward), requires_grad=False)
        if self.cuda:
            reward.cuda()
        values, _ = self.values(state).max(dim=1, keepdim=True)
        return reward + gamma * values


class DQN(nn.Module):
    def __init__(self, state_size, num_actions):
        super(DQN, self).__init__()

        self.state_size = state_size
        self.num_actions = num_actions

        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(state_size)
        self.conv1 = nn.Conv2d(state_size, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.bn0(x)
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
