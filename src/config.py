import torch

EPSILON_START = 1.0
EPSILON_FINAL = 0.1
FINAL_EXPLORATION_STEP = 1_250_000  # Exploration ends at 1.25M env steps

BATCH_SIZE = 32
REPLAY_START_SIZE = 50_000  # Minimum replay memory size before training starts
GAMMA = 0.99
TARGET_UPDATE_FREQ = 10_000  # policy_net weights copied to target_net every 10,000 env steps
LR = 0.00025
MEMORY_CAPACITY = 1_000_000
MAX_STEPS = 10_000_000  # ~40M ALE frames (40M / frame_skip=4)
TOTAL_STEPS = MAX_STEPS  # buffer filling is outside the training loop
EVAL_FREQ = 10_000   # Evaluate every 10,000 env steps
HELD_OUT_SIZE = 2_000  # Number of states for avg Q-value tracking

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
