# DQN Breakout - From Scratch

A PyTorch implementation of the Deep Q-Network (DQN) agent for the Atari game **Breakout**, built from scratch without RL libraries. This project follows the architecture and preprocessing standards established in the 2013 Nature paper, *Playing Atari with Deep Reinforcement Learning*.

---

## 🚀 Getting Started

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- NVIDIA GPU with CUDA (recommended for 22M frame training)

### Installation
1. Create the environment:
   ```bash
   conda create -n dqn_breakout python=3.10 -y
   conda activate dqn_breakout
   ```
2. Install dependencies:
   ```bash
   pip install gymnasium[atari,accept-rom-license] ale-py torch torchvision torchaudio matplotlib pandas opencv-python
   ```

### Running Training
To start a new training run:
```bash
python train.py
```

To resume from a specific checkpoint:
```bash
python train.py --resume logs/<folder>/dqn_breakout.pth --frame <last_frame>
```

---

## 🛠 Project Requirements

- **Environment:** `BreakoutNoFrameskip-v4` via Gymnasium/ALE.
- **Architecture:** 
  - 2 Convolutional layers (8x8 stride 4, 4x4 stride 2)
  - 1 Fully Connected layer (256 units)
- **Preprocessing:**
  - Grayscale conversion & 84x84 cropping.
  - Frame stacking (last 4 frames) to capture motion.
  - Reward clipping to `{-1, 0, +1}` for stability.
  - Termination on life loss (training signal).
- **Hyperparameters:**
  - Optimizer: RMSProp
  - Replay Memory: 1,000,000 capacity
  - Epsilon Schedule: Two-phase linear decay (0.147 -> 0.1 -> 0.01)
  - Target Network Update: Every 2,500 steps.

---

## 📁 Project Structure
- `train.py`: Main execution loop and checkpoint management.
- `agent.py`: DQN logic, Replay Memory, and Epsilon-Greedy policy.
- `model.py`: PyTorch CNN architecture.
- `preprocessing.py`: Custom Gymnasium wrappers for Atari frame processing.
- `cartpole/`: A lightweight test suite for algorithm verification.
