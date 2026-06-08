"""
Generate evaluation report charts and summary table from eval_results JSON files.

Usage:
    python scripts/plot_eval.py --results logs/run3_job_13184140/eval_results1.json \
                                           logs/run3_job_13184140/eval_results2.json \
                                           logs/run3_job_13184140/eval_results3.json

Produces (in report/):
    - score_distribution.png  histogram with mean / median / IQM marked
    - score_boxplot.png        box plot with individual scores overlaid
    - summary_table.png        stats table (mean, std, median, IQM, min, max)
    - Prints summary to stdout

IQM (Agarwal et al. 2021): pool all scores across runs, sort, take the middle 50% by
index, return their mean. Interpolates between mean and median — robust to outliers
without discarding as much signal as the median.
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return json.load(f)


def iqm(scores):
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    return sorted_scores[lo:hi].mean()


def plot_histogram(scores, stats, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores, bins=15, color="#1f77b4", edgecolor="white", alpha=0.85)

    ax.axvline(stats["mean"],   color="#d62728", linewidth=1.8, linestyle="-",  label=f"Mean   {stats['mean']:.1f}")
    ax.axvline(stats["median"], color="#ff7f0e", linewidth=1.8, linestyle="--", label=f"Median {stats['median']:.1f}")
    ax.axvline(stats["iqm"],    color="#2ca02c", linewidth=1.8, linestyle=":",  label=f"IQM    {stats['iqm']:.1f}")

    ax.set_xlabel("Episode Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution — {stats['n_games']} games (ε={stats['epsilon']})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_boxplot(scores, stats, out_path):
    fig, ax = plt.subplots(figsize=(5, 7))

    ax.boxplot(scores, patch_artist=True, widths=0.4,
               boxprops=dict(facecolor="#1f77b4", alpha=0.6),
               medianprops=dict(color="#ff7f0e", linewidth=2),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5),
               flierprops=dict(marker="o", markersize=5, alpha=0.5))

    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(scores))
    ax.scatter(1 + jitter, scores, color="#1f77b4", alpha=0.5, s=20, zorder=3)

    ax.axhline(stats["mean"], color="#d62728", linewidth=1.5, linestyle="--", label=f"Mean {stats['mean']:.1f}")
    ax.axhline(stats["iqm"],  color="#2ca02c", linewidth=1.5, linestyle=":",  label=f"IQM  {stats['iqm']:.1f}")

    ax.set_xticks([])
    ax.set_ylabel("Episode Score")
    ax.set_title(f"Score Box Plot — {stats['n_games']} games")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_table(stats, out_path):
    rows = [
        ["Mean",   f"{stats['mean']:.2f}"],
        ["Std",    f"{stats['std']:.2f}"],
        ["Median", f"{stats['median']:.2f}"],
        ["IQM",    f"{stats['iqm']:.2f}"],
        ["Min",    f"{stats['min']}"],
        ["Max",    f"{stats['max']}"],
    ]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.4, 1.6)

    for row_idx in [0, 3]:
        for col_idx in range(2):
            tbl[row_idx + 1, col_idx].set_facecolor("#dce8f5")

    ax.set_title(f"Evaluation Summary — {stats['n_games']} games (ε={stats['epsilon']})",
                 fontsize=10, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True, help="Paths to eval_results JSON files")
    args = parser.parse_args()

    all_scores = []
    epsilon = None
    for path in args.results:
        data = load(path)
        all_scores.extend(data["scores"])
        epsilon = data["epsilon"]

    scores = np.array(all_scores)

    stats = {
        "mean":    scores.mean(),
        "std":     scores.std(),
        "median":  float(np.median(scores)),
        "iqm":     iqm(scores),
        "min":     int(scores.min()),
        "max":     int(scores.max()),
        "n_games": len(scores),
        "epsilon": epsilon,
    }

    os.makedirs("report", exist_ok=True)

    plot_histogram(scores, stats, "report/score_distribution.png")
    plot_boxplot(scores,   stats, "report/score_boxplot.png")
    plot_table(stats,             "report/summary_table.png")

    print(f"{'Metric':<12} {'Value':>8}")
    print("-" * 22)
    for k, v in stats.items():
        if k in ("n_games", "epsilon"):
            continue
        print(f"{k.capitalize():<12} {v:>8.2f}" if isinstance(v, float) else f"{k.capitalize():<12} {v:>8}")
    print(f"\nPooled {stats['n_games']} games from {len(args.results)} eval run(s)")
    print("Saved plots to report/")


if __name__ == "__main__":
    main()
