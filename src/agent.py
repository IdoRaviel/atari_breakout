import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from model import DQN
from config import FINAL_EXPLORATION_STEP, EPSILON_START, EPSILON_FINAL


class ReplayMemory:
    """
    Circular replay buffer storing up to `capacity` transitions.

    Memory layout (all pre-allocated as contiguous numpy arrays):
      states      : (N, 4, 84, 84) uint8  — the 4-frame stack the agent observed
      next_frames : (N,    84, 84) uint8  — only the NEW frame that arrived after the action
      actions     : (N,)           int64
      rewards     : (N,)           float32
      dones       : (N,)           uint8   — 1 if the episode ended after this transition

    Why store only next_frames instead of the full next_state (4, 84, 84)?
      next_state shares 3 frames with state (frames 1-3 of state == frames 0-2 of next_state).
      Storing the full next_state would duplicate 75% of the data and triple memory usage
      for the state arrays. Instead, next_state is reconstructed at sample time by dropping
      the oldest frame from state and appending next_frames:
          next_state = [state[1], state[2], state[3], next_frame]
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.pos = 0   # write pointer (wraps around when buffer is full)
        self.size = 0  # number of valid transitions currently stored
        self.states      = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.next_frames = np.zeros((capacity, 84, 84),    dtype=np.uint8)
        self.actions     = np.zeros((capacity,),            dtype=np.int64)
        self.rewards     = np.zeros((capacity,),            dtype=np.float32)
        self.dones       = np.zeros((capacity,),            dtype=np.uint8)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos]      = state           # full 4-frame stack, already uint8
        self.next_frames[self.pos] = next_state[-1]  # only the newest frame (frame index 3)
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.dones[self.pos]       = done
        self.pos  = (self.pos + 1) % self.capacity   # advance write pointer, wrap at capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        # Normalize to float32 [0, 1] at sample time to keep stored data compact (uint8)
        states      = self.states[idx].astype(np.float32) / 255.0
        next_frames = self.next_frames[idx].astype(np.float32) / 255.0

        # Reconstruct next_state: drop oldest frame (index 0), append the new frame
        # states[:, 1:] has shape (B, 3, 84, 84); next_frames[:, np.newaxis] is (B, 1, 84, 84)
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

        # policy_net: updated every gradient step; used for action selection
        # target_net: frozen copy; updated every target_update_freq gradient steps
        # Keeping the target fixed breaks the otherwise moving-target problem in the
        # Bellman backup, stabilizing training.
        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), lr=lr, alpha=0.95, eps=0.01
        )
        self.memory = ReplayMemory(memory_capacity)

        self.steps_done = 0   # env steps, set externally by train.py
        self.update_count = 0 # gradient updates, used to trigger target net sync

    def get_epsilon(self):
        step = self.steps_done
        if step >= FINAL_EXPLORATION_STEP:
            return EPSILON_FINAL
        return EPSILON_START - (EPSILON_START - EPSILON_FINAL) * (step / FINAL_EXPLORATION_STEP)

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

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states      = torch.from_numpy(states).to(self.device, non_blocking=True)
        actions     = torch.from_numpy(actions).to(self.device, non_blocking=True)
        rewards     = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        dones       = torch.from_numpy(dones).to(self.device, non_blocking=True)

        # Q(s, a): value the policy network assigns to the action that was actually taken
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: r + γ * max_a' Q_target(s', a')
        # done=1 masks out the future-return term for terminal transitions.
        # no_grad: the target is treated as a fixed label — gradients only flow through current_q.
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones.float())

        # Huber loss (smooth_l1): behaves like MSE near zero, L1 for large errors,
        # which clips the gradient magnitude and stabilizes training with outlier rewards.
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically copy policy_net weights into target_net (hard update)
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
