import torch

EPSILON_START = 1.0
EPSILON_FINAL = 0.1
# Epsilon decay period: 8,000,000 raw ALE frames (extended for shaped-reward run).
# With action repeat=4: 8,000,000 / 4 = 2,000,000 env steps (policy decisions).
# With update frequency=4: 2,000,000 / 4 = 500,000 gradient updates during decay.
# Exploration is measured by env interaction, not gradient updates — so we set
# FINAL_EXPLORATION_STEP in env steps (policy decisions), not update count.
FINAL_EXPLORATION_STEP = 2_000_000

UPDATE_FREQ = 4  # gradient updates every 4 env steps = every 16 raw ALE frames
BATCH_SIZE = 32
REPLAY_START_SIZE = 50_000  # Minimum replay memory size before training starts
GAMMA = 0.99
# Target net synced every 10,000 gradient update steps (assignment spec + paper).
# Counted in gradient updates, not env steps — matches EVAL_FREQ unit below.
# With update frequency=4: 10,000 gradient updates × 4 env steps/update = 40,000 env steps.
TARGET_UPDATE_FREQ = 10_000  # gradient update steps
LR = 0.00025
MEMORY_CAPACITY = 1_000_000
MAX_STEPS = 25_000_000  # 100M raw ALE frames (100M / frame_skip=4)
TOTAL_STEPS = MAX_STEPS  # buffer filling is outside the training loop
# Eval every 10,000 gradient update steps = 40,000 env steps (assignment requirement).
# Counted in env steps here; divide by UPDATE_FREQ=4 to get gradient update steps.
EVAL_FREQ = 40_000  # env steps (= 10,000 gradient update steps)
HELD_OUT_SIZE = 2_000  # Number of states for avg Q-value tracking
NOOP_MAX = 9           # no-op actions on reset; each = 4 raw frames (4–36 raw frames total)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
