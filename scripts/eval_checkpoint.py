import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
from datetime import datetime

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import make_env
from model import build_model


def evaluate_checkpoint(model_path, n_games=30, epsilon=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_path, n_actions=4, device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint.get("step", "?")
    else:
        model.load_state_dict(checkpoint)
        step = "?"
    model.eval()
    print(f"Loaded checkpoint from step {step} | device: {device}")

    # Full game: terminal_on_life_loss=False so all 5 lives are played
    env = make_env(clip_reward=False, terminal_on_life_loss=False)

    scores = []
    for game in range(1, n_games + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                obs_t = torch.from_numpy(obs.astype(np.float32) / 255.0).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(obs_t).argmax(dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        scores.append(total_reward)
        print(f"  Game {game:3d}/{n_games}: {total_reward:.0f}")

    env.close()

    scores = np.array(scores)
    print(f"\nResults over {n_games} games:")
    print(f"  Mean:   {scores.mean():.1f}")
    print(f"  Std:    {scores.std():.1f}")
    print(f"  Min:    {scores.min():.0f}")
    print(f"  Max:    {scores.max():.0f}")
    print(f"  Median: {np.median(scores):.0f}")

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_step": step,
        "n_games": n_games,
        "epsilon": epsilon,
        "mean": round(float(scores.mean()), 2),
        "std": round(float(scores.std()), 2),
        "min": int(scores.min()),
        "max": int(scores.max()),
        "median": float(np.median(scores)),
        "scores": [float(s) for s in scores],
    }
    log_dir = os.path.dirname(os.path.abspath(model_path))
    out_path = os.path.join(log_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--games", type=int, default=30, help="Number of full games to evaluate")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Evaluation epsilon (default: 0.05)")
    args = parser.parse_args()

    if args.model:
        model_path = args.model
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, "..", "logs")
        pth_files = glob.glob(os.path.join(logs_dir, "**/*.pth"), recursive=True)
        if not pth_files:
            print("No .pth files found in logs/")
            exit(1)
        model_path = max(pth_files, key=os.path.getmtime)
        print(f"Using latest checkpoint: {model_path}")

    evaluate_checkpoint(model_path, n_games=args.games, epsilon=args.epsilon)
