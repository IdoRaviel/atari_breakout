import gymnasium as gym
import torch
import time
import numpy as np
from preprocessing import make_env
from model import DQN

def simulate(model_path):
    # 1. Create environment with human render mode
    # We use clip_reward=False for eval/sim to see real rewards
    env = make_env(render_mode="human", clip_reward=False)
    n_actions = env.action_space.n

    # 2. Setup the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(n_actions).to(device)

    # 3. Load the weights
    try:
        # Map to CPU if no CUDA, or vice versa
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model weights loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        return

    # 4. Run the simulation
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        print(f"Starting Episode {episode + 1}")
        while not done:
            # Action selection (Greedy)
            # Obs shape is (4, 84, 84), add batch dimension -> (1, 4, 84, 84)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
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
    # Point to your latest saved model
    LATEST_MODEL = "logs/20260413_121717/dqn_breakout.pth"
    simulate(LATEST_MODEL)
