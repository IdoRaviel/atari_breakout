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
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, n_actions, lr, gamma, target_update_freq, memory_capacity, batch_size, device):
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = device
        
        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)
        
        self.steps_done = 0

    def get_epsilon(self):
        # Epsilon Schedule (GEMINI.md)
        # Phase 1: 0.14712 -> 0.1 over steps 2 -> 1,000,000
        # Phase 2: 0.1 -> 0.01 over steps 1,000,000 -> 22,000,000
        
        step = self.steps_done
        if step < 2:
            return 0.14712368077870006
        elif step < 1_000_000:
            # Linear interpolation: Phase 1
            # y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            x1, y1 = 2, 0.14712368077870006
            x2, y2 = 1_000_000, 0.1
            return y1 + (y2 - y1) * (step - x1) / (x2 - x1)
        elif step < 22_000_000:
            # Linear interpolation: Phase 2
            x1, y1 = 1_000_000, 0.1
            x2, y2 = 22_000_000, 0.01
            return y1 + (y2 - y1) * (step - x1) / (x2 - x1)
        else:
            return 0.01

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
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.ByteTensor(dones).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            # y_j = r_j + gamma * max Q(s_{j+1}, a'; theta^-)
            # if done, y_j = r_j
            expected_q = rewards + self.gamma * next_q * (1 - dones)
            
        loss = nn.functional.mse_loss(current_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping is standard for Atari DQN but not explicitly in 2013 paper.
        # However, it helps stability. I'll omit it for "scratch" accuracy unless needed.
        self.optimizer.step()
        
        self.steps_done += 1
        
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
