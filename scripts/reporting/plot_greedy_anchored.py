"""Seed-level RL-minus-BC durations for the frozen inference comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PAIRED = ROOT / "results" / "tables" / "final" / "greedy_anchored_paired.csv"
OUT = ROOT / "results" / "final" / "figures" / "fig_greedy_anchored_deltas.png"


def main() -> None:
    rows = list(csv.DictReader(PAIRED.open(encoding="utf-8")))
    selected = [
        row for row in rows
        if row["temperature"] == "1.0" and row["metric"] in ("greedy_h", "anchored_h", "sample_mean_h")
    ]
    order = ["greedy_h", "anchored_h", "sample_mean_h"]
    labels = {
        "greedy_h": "Greedy\nreturned route",
        "anchored_h": "Greedy + 3 samples\nreturned route",
        "sample_mean_h": "Mean of 3 samples\nexpected draw",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    for x, metric in enumerate(order):
        row = next(item for item in selected if item["metric"] == metric)
        deltas = [float(value) for value in row["deltas_h"].split("|")]
        ax.scatter([x] * len(deltas), deltas, s=36, color="#1f4e79", zorder=3)
        ax.plot([x - 0.18, x + 0.18], [float(row["mean_h"])] * 2, color="#c45c26", lw=2, zorder=2)
    ax.axhline(0.0, color="#666666", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[metric] for metric in order])
    ax.set_ylabel("RL − BC duration (h)")
    ax.set_xlim(-0.5, 2.5)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
