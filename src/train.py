import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from preprocessing import make_env
from agent import DQNAgent
import argparse

# 2013 paper hyperparameters
EPSILON_START = 1.0
EPSILON_FINAL = 0.1
FINAL_EXPLORATION_STEP = 1_000_000

BATCH_SIZE = 32
REPLAY_START_SIZE = 50_000
GAMMA = 0.99
LR = 0.00025
MEMORY_CAPACITY = 1_000_000
MAX_STEPS = 12_500_000  # ~50M ALE frames (50M / frame_skip=4)
TOTAL_STEPS = MAX_STEPS + REPLAY_START_SIZE
EVAL_FREQ = 10_000
HELD_OUT_SIZE = 10_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(agent, n_episodes=3):
    env = make_env(clip_reward=False)
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
    states = torch.FloatTensor(held_out_states).to(agent.device)
    with torch.no_grad():
        q_values = agent.policy_net(states)
        max_q = q_values.max(dim=1)[0]
    return max_q.mean().item()


def save_run_config(log_dir, run_number=None):
    config = {
        "run_number": run_number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "paper": "Mnih et al. 2013 (NIPS)",
        "game": "BreakoutNoFrameskip-v4",
        "hyperparameters": {
            "epsilon_start": EPSILON_START,
            "epsilon_final": EPSILON_FINAL,
            "final_exploration_step": FINAL_EXPLORATION_STEP,
            "batch_size": BATCH_SIZE,
            "replay_start_size": REPLAY_START_SIZE,
            "gamma": GAMMA,
            "learning_rate": LR,
            "memory_capacity": MEMORY_CAPACITY,
            "max_steps": MAX_STEPS,
            "total_steps": TOTAL_STEPS,
        },
        "architecture": {
            "input_shape": [4, 84, 84],
            "conv1": "16 filters, 8x8, stride 4",
            "conv2": "32 filters, 4x4, stride 2",
            "fc": "256 units",
            "optimizer": "RMSprop(lr=0.00025, alpha=0.95, eps=0.01)",
            "loss": "MSE",
            "target_network": False,
        },
        "preprocessing": {
            "frame_skip": 4,
            "frame_stack": 4,
            "grayscale": True,
            "resize": "110x84 -> crop 84x84",
            "normalize": "divide by 255",
            "reward_shaping": "clip to [-1, 1]",
            "life_loss_as_terminal": True,
        },
        "device": str(DEVICE),
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def train(resume_path=None, start_frame=1, run_number=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"run{run_number}_" if run_number else ""
    log_dir = os.path.join("logs", f"{run_label}{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    save_run_config(log_dir, run_number)

    env = make_env()
    n_actions = env.action_space.n
    agent = DQNAgent(
        n_actions=n_actions,
        lr=LR,
        gamma=GAMMA,
        memory_capacity=MEMORY_CAPACITY,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        agent.policy_net.load_state_dict(torch.load(resume_path, map_location=DEVICE))
        agent.steps_done = start_frame

    stats = []
    stats_path = (
        os.path.join(os.path.dirname(resume_path), "training_stats.csv")
        if resume_path else None
    )
    if stats_path and os.path.exists(stats_path):
        try:
            old_stats = pd.read_csv(stats_path)
            stats = old_stats.to_dict("records")
            print(f"Loaded {len(stats)} previous stats entries.")
        except Exception as e:
            print(f"Could not load previous stats: {e}")

    obs, info = env.reset()
    print(f"Starting training on {DEVICE} from step {start_frame}...")

    held_out_states = None

    for step_idx in range(start_frame, TOTAL_STEPS + 1):
        agent.steps_done = step_idx
        epsilon = agent.get_epsilon()
        action = agent.select_action(obs, epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.memory.push(obs, action, reward, next_obs, done)
        obs = next_obs

        if len(agent.memory) >= REPLAY_START_SIZE:
            if held_out_states is None:
                states, _, _, _, _ = agent.memory.sample(HELD_OUT_SIZE)
                held_out_states = states
                print(f"Collected {HELD_OUT_SIZE} held-out states for Q-value tracking.")

            agent.update()

        if done:
            obs, info = env.reset()

        if step_idx % EVAL_FREQ == 0:
            avg_reward = evaluate(agent)
            avg_q = compute_avg_q(agent, held_out_states) if held_out_states is not None else None
            print(
                f"Step {step_idx}: Epsilon {epsilon:.4f}, Eval Reward: {avg_reward:.2f}, Avg Q: {avg_q:.4f}"
                if avg_q is not None
                else f"Step {step_idx}: Epsilon {epsilon:.4f}, Eval Reward: {avg_reward:.2f}"
            )
            stats.append({"step": step_idx, "reward": avg_reward, "avg_q": avg_q})

            df = pd.DataFrame(stats)
            df.to_csv(os.path.join(log_dir, "training_stats.csv"), index=False)
            torch.save(agent.policy_net.state_dict(), os.path.join(log_dir, "dqn_breakout.pth"))

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--run", type=int, default=None)
    args = parser.parse_args()

    train(resume_path=args.resume, start_frame=args.frame, run_number=args.run)
