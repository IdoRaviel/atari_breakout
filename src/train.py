import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from preprocessing import make_env
from agent import DQNAgent
import argparse

# Hyperparameters from GEMINI.md
EPSILON_START = 0.14712368077870006
EPSILON_KNEE = 0.1
EPSILON_FINAL = 0.01
KNEE_STEP = 1_000_000
FINAL_STEP = 22_000_000

BATCH_SIZE = 84
REPLAY_START_SIZE = 50_000
GAMMA = 0.9529810958690669
TARGET_UPDATE_FREQ = 2500
LR = 0.0000665660642462102
MEMORY_CAPACITY = 1_000_000
MAX_FRAMES = 22_000_000
EVAL_FREQ = 10_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(agent, n_episodes=5):
    """
    Evaluate the agent with near-greedy policy (epsilon = 0.05).
    """
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

def train(resume_path=None, start_frame=1):
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    env = make_env()
    n_actions = env.action_space.n
    agent = DQNAgent(
        n_actions=n_actions,
        lr=LR,
        gamma=GAMMA,
        target_update_freq=TARGET_UPDATE_FREQ,
        memory_capacity=MEMORY_CAPACITY,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        agent.policy_net.load_state_dict(torch.load(resume_path, map_location=DEVICE))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.steps_done = start_frame
    
    stats = []
    # If resuming, we might want to load previous stats too
    stats_path = os.path.join(os.path.dirname(resume_path), "training_stats.csv") if resume_path else None
    if stats_path and os.path.exists(stats_path):
        try:
            old_stats = pd.read_csv(stats_path)
            stats = old_stats.to_dict('records')
            print(f"Loaded {len(stats)} previous stats entries.")
        except Exception as e:
            print(f"Could not load previous stats: {e}")

    obs, info = env.reset()
    
    print(f"Starting training on {DEVICE} from frame {start_frame}...")
    
    for frame_idx in range(start_frame, MAX_FRAMES + 1):
        agent.steps_done = frame_idx # Sync for epsilon schedule
        epsilon = agent.get_epsilon()
        action = agent.select_action(obs, epsilon)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.memory.push(obs, action, reward, next_obs, done)
        obs = next_obs
        
        if len(agent.memory) >= REPLAY_START_SIZE:
            loss = agent.update()
        else:
            loss = None
            
        if done:
            obs, info = env.reset()
            
        if frame_idx % EVAL_FREQ == 0:
            avg_reward = evaluate(agent)
            print(f"Frame {frame_idx}: Epsilon {epsilon:.4f}, Eval Reward: {avg_reward:.2f}")
            stats.append({'frame': frame_idx, 'reward': avg_reward})
            
            # Save stats to log directory
            df = pd.DataFrame(stats)
            df.to_csv(os.path.join(log_dir, "training_stats.csv"), index=False)
            
            # Save model to log directory
            torch.save(agent.policy_net.state_dict(), os.path.join(log_dir, "dqn_breakout.pth"))

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to checkpoint .pth file", default=None)
    parser.add_argument("--frame", type=int, help="Frame to start from", default=1)
    args = parser.parse_args()
    
    train(resume_path=args.resume, start_frame=args.frame)
