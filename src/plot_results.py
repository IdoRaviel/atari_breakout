"""
Generate report graphs from training_stats.csv files.

Usage:
    python plot_results.py --runs logs/run1_<ts> logs/run2_<ts> logs/run3_<ts>

Produces:
    - report/learning_curves.png  (all 3 runs on one axes)
    - report/run1.png, run2.png, run3.png  (individual plots)
    - Prints final rewards and average to stdout
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def load(run_dir):
    path = os.path.join(run_dir, "training_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No training_stats.csv in {run_dir}")
    return pd.read_csv(path)


def plot_individual(df, run_label, color, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["step"], df["reward"], color=color, linewidth=1.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Evaluation Reward")
    ax.set_title(f"Learning Curve — {run_label}")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1e6)}M"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined(dfs, labels, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for df, label, color in zip(dfs, labels, COLORS):
        ax.plot(df["step"], df["reward"], label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Evaluation Reward")
    ax.set_title("Learning Curves — All Runs (Breakout, BreakoutNoFrameskip-v4)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1e6)}M"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Paths to log dirs")
    args = parser.parse_args()

    if len(args.runs) != 3:
        raise ValueError(f"Expected 3 run directories, got {len(args.runs)}")

    os.makedirs("report", exist_ok=True)

    dfs = []
    labels = []
    final_rewards = []

    for i, run_dir in enumerate(args.runs, start=1):
        label = f"Run {i}"
        df = load(run_dir)
        dfs.append(df)
        labels.append(label)

        final_reward = df["reward"].iloc[-1]
        final_rewards.append(final_reward)

        plot_individual(
            df, label, COLORS[i - 1], os.path.join("report", f"run{i}.png")
        )
        print(f"{label}: {len(df)} eval points, "
              f"{int(df['step'].iloc[-1]):,} total steps, "
              f"final reward = {final_reward:.2f}")

    plot_combined(dfs, labels, os.path.join("report", "learning_curves.png"))

    avg = sum(final_rewards) / len(final_rewards)
    print(f"\nFinal rewards : {[round(r, 2) for r in final_rewards]}")
    print(f"Average       : {avg:.2f}")
    print("\nSaved plots to report/")


if __name__ == "__main__":
    main()
