import os
import sys
import glob
import time
import random
import argparse
import numpy as np
import torch

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import make_env
from model import build_model


def simulate(model_path):
    # 1. Create environment with human render mode
    # We use clip_reward=False for eval/sim to see real rewards
    env = make_env(render_mode="human", clip_reward=False, terminal_on_life_loss=False)
    n_actions = env.action_space.n

    # 2. Setup the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_path, n_actions=n_actions, device=device)

    # 3. Load the weights
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model weights loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        return

    # 4. Run the simulation
    for episode in range(3):
        obs, info = env.reset()  # FireResetEnv auto-presses FIRE on reset
        episode_reward = 0
        done = False

        print(f"Starting Episode {episode + 1}")
        while not done:
            # Action selection (epsilon-greedy, epsilon=0.05)
            # Obs shape is (4, 84, 84), add batch dimension -> (1, 4, 84, 84)
            if random.random() < 0.05:
                action = env.action_space.sample()
            else:
                obs_tensor = torch.from_numpy(obs.astype(np.float32) / 255.0).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(obs_tensor).argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            # Slow down slightly for human viewing
            time.sleep(0.01)

        print(f"Episode {episode + 1} finished with total reward: {episode_reward}")
        time.sleep(1)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to .pth file")
    args = parser.parse_args()

    if args.model:
        simulate(args.model)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, "..", "logs")
        pth_files = glob.glob(os.path.join(logs_dir, "**/*.pth"), recursive=True)
        if not pth_files:
            print("No .pth files found in logs/")
            exit(1)
        LATEST_MODEL = max(pth_files, key=os.path.getmtime)
        print(f"Using model: {LATEST_MODEL}")
        simulate(LATEST_MODEL)
