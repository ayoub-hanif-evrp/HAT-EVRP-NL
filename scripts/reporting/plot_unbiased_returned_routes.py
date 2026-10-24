"""Seed-level returned-route differences on the unbiased splits."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PAIRED = ROOT / "results" / "tables" / "final" / "unbiased_eval_paired.csv"
OBJECTIVE = ROOT / "results" / "tables" / "final" / "objective_aligned_paired.csv"
OUT = ROOT / "results" / "final" / "figures" / "fig_unbiased_returned_routes.png"


def _row(path: Path, split: str, candidate: str):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return next(
        row for row in rows
        if row["split"] == split and row["candidate"] == candidate and row["metric"] == "anchored_h"
    )


def main() -> None:
    panels = [
        ("dev_v2", "Development"),
        ("confirm_v2", "Confirmation"),
        ("shift_v2c", "Montoya-like"),
        ("shift_slow", "Slow only"),
        ("shift_geo", "Geography"),
    ]
    series = [
        ("original_rl", PAIRED, "#1f4e79", "Original RL"),
        ("greedy_baseline_rl", PAIRED, "#5b8e7d", "Greedy-baseline RL"),
        ("objective_aligned_rl", OBJECTIVE, "#c45c26", "Best-of-K RL"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=200)
    for x, (split, _label) in enumerate(panels):
        for offset, (candidate, path, color, _name) in enumerate(series):
            row = _row(path, split, candidate)
            deltas = [float(value) for value in row["deltas_h"].split("|")]
            xs = [x + (offset - 1) * 0.18] * len(deltas)
            ax.scatter(xs, deltas, s=22, color=color, zorder=3)
    ax.axhline(0.0, color="#666666", lw=0.8)
    ax.set_xticks(range(len(panels)))
    ax.set_xticklabels([label for _split, label in panels])
    ax.set_ylabel("RL − BC penalized duration (h)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=6, label=name)
        for _candidate, _path, color, name in series
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
