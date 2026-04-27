import torch

EPSILON_START = 1.0
EPSILON_FINAL = 0.1
# Epsilon decay period: 1,000,000 raw ALE frames (paper).
# With action repeat=4: 1,000,000 / 4 = 250,000 env steps (policy decisions).
# With update frequency=4: 250,000 / 4 = 62,500 gradient updates during decay.
# Exploration is measured by env interaction, not gradient updates — so we set
# FINAL_EXPLORATION_STEP in env steps (policy decisions), not update count.
FINAL_EXPLORATION_STEP = 250_000

UPDATE_FREQ = 4  # gradient updates every 4 env steps = every 16 raw ALE frames
BATCH_SIZE = 32
REPLAY_START_SIZE = 50_000  # Minimum replay memory size before training starts
GAMMA = 0.99
# Target net synced every 10,000 gradient updates (paper).
# With update frequency=4: 10,000 updates × 4 env steps/update = 40,000 env steps.
TARGET_UPDATE_FREQ = 10_000
LR = 0.00025
MEMORY_CAPACITY = 1_000_000
MAX_STEPS = 12_500_000  # 50M ALE frames (50M / frame_skip=4), matching 2015 paper
TOTAL_STEPS = MAX_STEPS  # buffer filling is outside the training loop
# Eval every 10,000 gradient updates = 40,000 env steps (matching assignment requirement).
# Paper evaluates every 50,000 gradient updates; we are 5x more frequent.
EVAL_FREQ = 40_000
HELD_OUT_SIZE = 2_000  # Number of states for avg Q-value tracking

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
