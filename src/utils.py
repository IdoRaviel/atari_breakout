import os
import json
import numpy as np
import torch
from datetime import datetime
from preprocessing import make_env
from config import (
    EPSILON_START, EPSILON_FINAL, FINAL_EXPLORATION_STEP,
    BATCH_SIZE, REPLAY_START_SIZE, GAMMA, TARGET_UPDATE_FREQ,
    LR, MEMORY_CAPACITY, MAX_STEPS, TOTAL_STEPS, EVAL_FREQ, HELD_OUT_SIZE, DEVICE,
    UPDATE_FREQ, NOOP_MAX,
)


def fill_replay_buffer(memory, env, replay_start_size):
    """Fill replay buffer with random actions before training starts."""
    print(f"Filling replay buffer ({replay_start_size} samples)...")
    obs, _ = env.reset()
    while len(memory) < replay_start_size:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()
    print("Buffer ready.")


def evaluate(agent, n_episodes=1):
    """
    Evaluate the agent over n full games (all 5 lives each) with epsilon=0.05.
    terminal_on_life_loss=False: life loss auto-injects FIRE and continues;
    episode only ends on true game over, matching the paper's reported scores.
    """
    env = make_env(clip_reward=False, terminal_on_life_loss=False)
    total_rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(obs, epsilon=0.05)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)
    env.close()
    return np.mean(total_rewards)


def compute_avg_q(agent, held_out_states):
    """
    Compute the average max Q-value over a fixed held-out set of states.
    Tracks learning progress without the noise of episode rewards.
    """
    # held_out_states is already a GPU tensor — no transfer needed
    with torch.no_grad():
        q_values = agent.policy_net(held_out_states)  # (N, n_actions)
        max_q = q_values.max(dim=1)[0]  # (N,)
    return max_q.mean().item()


def save_run_config(log_dir, run_number=None):
    config = {
        "run_number": run_number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "game": "BreakoutNoFrameskip-v4",
        "hyperparameters": {
            "epsilon_start": EPSILON_START,
            "epsilon_final": EPSILON_FINAL,
            "final_exploration_step": FINAL_EXPLORATION_STEP,
            "batch_size": BATCH_SIZE,
            "replay_start_size": REPLAY_START_SIZE,
            "gamma": GAMMA,
            "target_update_freq": TARGET_UPDATE_FREQ,
            "learning_rate": LR,
            "memory_capacity": MEMORY_CAPACITY,
            "max_steps": MAX_STEPS,
            "total_steps": TOTAL_STEPS,
        },
        "architecture": {
            "input_shape": [4, 84, 84],
            "conv1": "32 filters, 8x8, stride 4",
            "conv2": "64 filters, 4x4, stride 2",
            "conv3": "64 filters, 3x3, stride 1",
            "fc": "512 units",
            "optimizer": "RMSprop(lr=0.00025, alpha=0.95, eps=0.01)",
            "loss": "Huber (smooth_l1)",
        },
        "preprocessing": {
            "frame_skip": 4,
            "update_frequency": UPDATE_FREQ,
            "frame_stack": 4,
            "grayscale": True,
            "resize": "110x84 -> crop 84x84",
            "normalize": "divide by 255",
            "reward_shaping": "clip to [-1, 1]",
            "life_loss_as_terminal": True,
            "fire_reset": True,
            "noop_reset_max_agent_steps": NOOP_MAX,
        },
        "evaluation": {
            "eval_freq_steps": EVAL_FREQ,
            "eval_episodes": 1,
            "eval_epsilon": 0.05,
            "held_out_states": HELD_OUT_SIZE,
        },
        "device": str(DEVICE),
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
