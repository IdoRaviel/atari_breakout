import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from model import DQN


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.states      = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.next_frames = np.zeros((capacity, 84, 84),    dtype=np.uint8)
        self.actions     = np.zeros((capacity,),            dtype=np.int64)
        self.rewards     = np.zeros((capacity,),            dtype=np.float32)
        self.dones       = np.zeros((capacity,),            dtype=np.uint8)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos]      = state        # already uint8
        self.next_frames[self.pos] = next_state[-1]  # already uint8
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.dones[self.pos]       = done
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        states      = self.states[idx].astype(np.float32) / 255.0
        next_frames = self.next_frames[idx].astype(np.float32) / 255.0
        next_states = np.concatenate(
            [states[:, 1:], next_frames[:, np.newaxis]], axis=1
        )
        return (
            states,
            self.actions[idx],
            self.rewards[idx],
            next_states,
            self.dones[idx],
        )

    def __len__(self):
        return self.size


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
            state = torch.from_numpy(state.astype(np.float32) / 255.0).unsqueeze(0).to(self.device, non_blocking=True)
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        states      = torch.from_numpy(states).to(self.device, non_blocking=True)
        actions     = torch.from_numpy(actions).to(self.device, non_blocking=True)
        rewards     = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        dones       = torch.from_numpy(dones).to(self.device, non_blocking=True)
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
