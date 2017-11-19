import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()

        self.in_channels = in_channels
        self.num_actions = num_actions

        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.bn0(x)
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
