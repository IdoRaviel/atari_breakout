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

#### Running Evaluation

Evaluation requires a GPU node on the BIU Slurm cluster. Connect and request an interactive session:

```bash
ssh slurm-login1.lnx.biu.ac.il
srun --partition=generic --gres=gpu:1 --mem=16G --time=00:30:00 --pty bash
```

Then activate the environment and run:

```bash
conda activate dqn_breakout
python scripts/eval_checkpoint.py --model logs/<run_folder>/<checkpoint>.pth --games 30 --epsilon 0.05
```

Results are saved automatically to `logs/<run_folder>/eval_results.json`.

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
- `src/train.py`: Main execution loop and checkpoint management.
- `src/agent.py`: DQN logic, Replay Memory, and Epsilon-Greedy policy.
- `src/model.py`: PyTorch CNN architecture.
- `src/model_factory.py`: Loads the correct model architecture from a checkpoint.
- `src/preprocessing.py`: Custom Gymnasium wrappers for Atari frame processing.
- `src/config.py`: Hyperparameter configuration.
- `src/utils.py`: Shared utility functions.
- `src/test_run.py`: Lightweight test suite for algorithm verification.
- `scripts/eval_checkpoint.py`: Evaluates a saved checkpoint over N full games.
- `scripts/plot_results.py`: Plots evaluation reward curves from training logs.
- `scripts/simulate_atari.py`: Visualizes the agent playing in real time.
- `logs/`: Training run outputs — checkpoints, metrics, and evaluation results.