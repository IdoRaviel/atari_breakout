import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        # 2013 paper architecture: 16 filters 8x8/4, 32 filters 4x4/2, FC 256
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # After conv1: (16, 20, 20)
        # After conv2: (32, 9, 9)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)
