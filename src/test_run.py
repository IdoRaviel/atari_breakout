from train import train
import train as train_module

# Patch constants for a quick smoke test
train_module.MAX_FRAMES = 100
train_module.REPLAY_START_SIZE = 50
train_module.EVAL_FREQ = 50

if __name__ == "__main__":
    train()
