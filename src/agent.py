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
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, n_actions, lr, gamma, memory_capacity, batch_size, device):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.policy_net = DQN(n_actions).to(device)

        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), lr=lr, alpha=0.95, eps=0.01
        )
        self.memory = ReplayMemory(memory_capacity)

        self.steps_done = 0

    def get_epsilon(self):
        step = self.steps_done
        if step >= 1_000_000:
            return 0.1
        return 1.0 - 0.9 * (step / 1_000_000)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state).argmax(dim=1).item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.ByteTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.policy_net(next_states).max(dim=1)[0]
            expected_q = rewards + self.gamma * next_q * (1 - dones.float())

        loss = nn.functional.mse_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
