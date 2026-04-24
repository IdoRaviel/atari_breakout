import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model import DQN


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            (state * 255).astype(np.uint8),
            action, reward,
            (next_state * 255).astype(np.uint8),
            done,
        ))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(state, dtype=np.float32) / 255.0,
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32) / 255.0,
            np.array(done, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        n_actions,
        lr,
        gamma,
        target_update_freq,
        memory_capacity,
        batch_size,
        device,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = device

        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 2015 paper: RMSprop with alpha=0.95, eps=0.01
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), lr=lr, alpha=0.95, eps=0.01
        )
        self.memory = ReplayMemory(memory_capacity)

        self.steps_done = 0  # env steps, set externally by train.py
        self.update_count = 0  # gradient updates, for target network sync

    def get_epsilon(self):
        # 2015 Nature paper: linear decay 1.0 -> 0.1 over first 1M env steps
        step = self.steps_done
        if step >= 1_000_000:
            return 0.1
        return 1.0 - (1.0 - 0.1) * (step / 1_000_000)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        # Extracting the tensors and moving to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.ByteTensor(dones).to(self.device)
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Compute target Q values using the target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            expected_q = rewards + self.gamma * next_q * (1 - dones.float())

        # clip TD error to [-1, 1] == Huber loss
        loss = nn.functional.smooth_l1_loss(current_q, expected_q)
        # Knows to update the policy network because current_q is derived from it, and expected_q is treated as a constant (no_grad)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
