import os
import json
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """2015 Nature DQN architecture. Input shape: (4, 84, 84)."""
    def __init__(self, n_actions):
        super().__init__()
        # After conv1: (32, 20, 20)
        # After conv2: (64, 9, 9)
        # After conv3: (64, 7, 7) -> flatten: 3136
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


class DQN2013(nn.Module):
    """2013 DQN architecture: 2 conv layers (16/32 filters), FC 512."""
    def __init__(self, n_actions):
        super().__init__()
        # After conv1: (16, 20, 20)
        # After conv2: (32, 9, 9) -> flatten: 2592
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)


def build_model(model_path, n_actions, device):
    """
    Read config.json from the checkpoint's log directory, detect the architecture,
    and return the appropriate model on the given device.
    Defaults to 2015 if no config.json is found.
    """
    log_dir = os.path.dirname(os.path.abspath(model_path))
    config_path = os.path.join(log_dir, "config.json")

    arch = "2015"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        if "conv3" not in config.get("architecture", {}):
            arch = "2013"

    print(f"Detected architecture: {arch}")
    if arch == "2015":
        return DQN(n_actions).to(device)
    else:
        return DQN2013(n_actions).to(device)
