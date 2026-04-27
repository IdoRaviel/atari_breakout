import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        # 2015 Nature DQN architecture. Input shape: (4, 84, 84)
        # After conv1: (32, (84-8)/4 + 1) = (32, 20, 20)
        # After conv2: (64, (20-4)/2 + 1) = (64, 9, 9)
        # After conv3: (64, (9-3)/1 + 1)  = (64, 7, 7) -> flatten: 3136
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)
