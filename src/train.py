import os
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from preprocessing import make_env
from agent import DQNAgent
from config import (
    LR, GAMMA, TARGET_UPDATE_FREQ, MEMORY_CAPACITY, BATCH_SIZE, DEVICE,
    REPLAY_START_SIZE, HELD_OUT_SIZE, TOTAL_STEPS, EVAL_FREQ, UPDATE_FREQ,
)
from utils import evaluate, compute_avg_q, save_run_config, fill_replay_buffer

# Always write logs to project root/logs, regardless of where the script is run from
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def train(resume_path=None, start_frame=1, run_number=None, log_dir_override=None):
    # --- Log directory ---
    # Resume: reuse the existing run's directory so stats and checkpoints stay together.
    # New run: create a timestamped subdirectory and write config.json.
    if resume_path and os.path.exists(resume_path):
        log_dir = os.path.dirname(resume_path)
        print(f"Resuming in existing log directory: {log_dir}")
    elif log_dir_override:
        log_dir = log_dir_override
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logging to: {log_dir}")
        save_run_config(log_dir, run_number)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_label = f"run{run_number}_" if run_number else ""
        log_dir = os.path.join(LOGS_DIR, f"{run_label}{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logging to: {log_dir}")
        save_run_config(log_dir, run_number)

    env = make_env(terminal_on_life_loss=False)
    n_actions = env.action_space.n
    agent = DQNAgent(
        n_actions=n_actions,
        lr=LR,
        gamma=GAMMA,
        target_update_freq=TARGET_UPDATE_FREQ,
        memory_capacity=MEMORY_CAPACITY,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    held_out_states = None
    held_out_uint8 = None

    # --- Checkpoint restore ---
    # Restores weights, optimizer state (momentum/variance accumulators), gradient
    # update count (for target net sync), and the held-out states (so Q tracking
    # is consistent across resume boundaries).
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=True)
        if "model_state_dict" in checkpoint:
            agent.policy_net.load_state_dict(checkpoint["model_state_dict"])
            agent.update_count = checkpoint.get("update_count", 0)
            if "optimizer_state_dict" in checkpoint:
                agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Restored optimizer state from checkpoint.")
            if "held_out_states" in checkpoint:
                held_out_uint8 = checkpoint["held_out_states"].cpu().numpy()
                held_out_states = torch.from_numpy(held_out_uint8.astype(np.float32) / 255.0).to(DEVICE)
                print(f"Restored {HELD_OUT_SIZE} held-out states from checkpoint.")
        else:
            agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.steps_done = start_frame

    # --- Stats CSV ---
    # Appended every EVAL_FREQ steps. Loaded on resume so the full training curve
    # is preserved in one file across multiple runs.
    stats = []
    stats_csv = os.path.join(log_dir, "training_stats.csv")
    if os.path.exists(stats_csv):
        try:
            stats = pd.read_csv(stats_csv).to_dict("records")
            print(f"Loaded {len(stats)} previous stats entries.")
        except Exception as e:
            print(f"Could not load previous stats: {e}")

    # --- Replay buffer warm-up ---
    # Fill the buffer with random transitions before training starts so the first
    # batches are not all correlated with the initial state distribution.
    if len(agent.memory) == 0:
        fill_replay_buffer(agent.memory, env, REPLAY_START_SIZE)

    # --- Held-out states for Q tracking ---
    # A fixed set of states sampled once from the replay buffer and never updated.
    # avg max-Q over this set is a low-noise proxy for learning progress because
    # it is not confounded by episode length variation the way reward is.
    if held_out_states is None:
        idx = np.random.randint(0, len(agent.memory), size=HELD_OUT_SIZE)
        held_out_uint8 = agent.memory.states[idx].copy()
        held_out_states = torch.from_numpy(held_out_uint8.astype(np.float32) / 255.0).to(DEVICE)
        print(f"Collected {HELD_OUT_SIZE} held-out states for Q-value tracking.")

    obs, info = env.reset()
    print(f"Starting training on {DEVICE} from step {start_frame}...")

    # --- Main training loop ---
    # Each iteration = 1 env step = 4 raw ALE frames (due to action repeat).
    # Total: TOTAL_STEPS env steps = TOTAL_STEPS * 4 raw frames.
    for step_idx in range(start_frame, TOTAL_STEPS + 1):

        # Epsilon decays linearly from 1.0 → 0.1 over FINAL_EXPLORATION_STEP env steps,
        # then stays fixed. steps_done is set here so the schedule is tied to env
        # interaction count, not gradient update count.
        agent.steps_done = step_idx
        epsilon = agent.get_epsilon()
        action = agent.select_action(obs, epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store the transition. done=True only on true game over (all 5 lives gone)
        # because terminal_on_life_loss=False; life losses auto-fire and continue.
        agent.memory.push(obs, action, reward, next_obs, done)
        obs = next_obs

        # Gradient update every UPDATE_FREQ env steps (paper: every 4 decisions = 16 raw frames).
        # This decouples acting frequency from learning frequency, reducing correlation
        # between consecutive training batches.
        if step_idx % UPDATE_FREQ == 0:
            agent.update()

        if done:
            obs, info = env.reset()

        # --- Periodic checkpoint + evaluation ---
        if step_idx % EVAL_FREQ == 0:
            # Save everything needed to resume: weights, optimizer, step counters,
            # and the held-out states so Q tracking stays consistent after resume.
            checkpoint_data = {
                "model_state_dict": agent.policy_net.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "step": step_idx,
                "update_count": agent.update_count,
            }
            if held_out_uint8 is not None:
                checkpoint_data["held_out_states"] = torch.from_numpy(held_out_uint8)
            torch.save(checkpoint_data, os.path.join(log_dir, "dqn_breakout.pth"))

            # Evaluate with epsilon=0.05 over 1 full game (5 lives, no life-loss terminal).
            # Single episode during training for speed; use eval_checkpoint.py for 30-game eval.
            avg_reward = evaluate(agent, n_episodes=1)
            avg_q = compute_avg_q(agent, held_out_states)
            print(f"Step {step_idx}: Epsilon {epsilon:.4f}, Eval Reward: {avg_reward:.2f}, Avg Q: {avg_q:.4f}")
            stats.append({"step": step_idx, "reward": avg_reward, "avg_q": avg_q})

            # Overwrite the CSV each time so a partial run always has a valid file.
            df = pd.DataFrame(stats)
            df.to_csv(stats_csv, index=False)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to checkpoint .pth file", default=None)
    parser.add_argument("--frame", type=int, help="Frame to start from", default=1)
    parser.add_argument("--run", type=int, help="Run number (1, 2, 3)", default=None)
    parser.add_argument("--logdir", type=str, help="Log directory path (overrides auto-generated name)", default=None)
    args = parser.parse_args()

    train(resume_path=args.resume, start_frame=args.frame, run_number=args.run, log_dir_override=args.logdir)
