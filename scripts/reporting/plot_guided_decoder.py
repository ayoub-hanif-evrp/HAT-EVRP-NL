"""Development figure for the guided decoder and the self-imitation control."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
GUIDED = ROOT / "results" / "tables" / "final" / "guided_decoder_paired.csv"
TRAIN = ROOT / "results" / "tables" / "final" / "self_imitation_paired.csv"
OUT = ROOT / "results" / "final" / "figures" / "fig_guided_decoder_development.png"
T_CRIT = float(stats.t.ppf(0.975, 4))


def _write_training_table() -> None:
    diagnostic = list(
        csv.DictReader(
            (ROOT / "results" / "raw" / "algorithmic_superiority_v1" / "guided_diagnostic" / "evaluations.csv").open(
                encoding="utf-8"
            )
        )
    )
    bc = {}
    for row in diagnostic:
        if row["policy"] == "bc" and row["physically_coverable"] == "1":
            bc.setdefault(row["seed"], []).append(float(row["greedy_h"]))
    bc_mean = {seed: sum(values) / len(values) for seed, values in bc.items()}
    root = ROOT / "results" / "raw" / "algorithmic_superiority_v1" / "self_imitation" / "runs"
    modes = {}
    for mode in ("imitation", "rl_imitation"):
        modes[mode] = {}
        for seed in bc_mean:
            meta = json.loads((root / mode / f"H2_seed{seed}" / "run_meta.json").read_text(encoding="utf-8"))
            modes[mode][seed] = float(meta["final_coverable_h"])
    rows = []
    for name, left, right in (
        ("imitation_minus_bc_greedy", bc_mean, modes["imitation"]),
        ("rl_imitation_minus_bc_greedy", bc_mean, modes["rl_imitation"]),
        ("rl_imitation_minus_imitation", modes["imitation"], modes["rl_imitation"]),
    ):
        seeds = sorted(left)
        deltas = [right[seed] - left[seed] for seed in seeds]
        array = np.asarray(deltas, dtype=float)
        mean = float(array.mean())
        half = T_CRIT * float(array.std(ddof=1)) / math.sqrt(len(array))
        relatives = [(right[seed] - left[seed]) / left[seed] for seed in seeds]
        nonzero = [value for value in deltas if value != 0.0]
        sign_p = (
            float(stats.binomtest(sum(value < 0 for value in nonzero), len(nonzero), 0.5, alternative="two-sided").pvalue)
            if nonzero
            else 1.0
        )
        wilcoxon_p = (
            1.0
            if np.allclose(array, 0.0)
            else float(stats.wilcoxon(array, alternative="two-sided", method="exact").pvalue)
        )
        rows.append(
            {
                "comparison": name,
                "subset": "coverable",
                "mean_h": mean,
                "ci_low_h": mean - half,
                "ci_high_h": mean + half,
                "n_negative": int(sum(value < 0 for value in deltas)),
                "mean_relative": sum(relatives) / len(relatives),
                "t_p": float(stats.ttest_1samp(array, 0.0).pvalue),
                "sign_p": sign_p,
                "wilcoxon_p": wilcoxon_p,
                "deltas_h": "|".join(f"{value:.6f}" for value in deltas),
            }
        )
    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _guided(comparison: str) -> dict:
    for row in csv.DictReader(GUIDED.open(encoding="utf-8")):
        if row["comparison"] == comparison and row["subset"] == "coverable":
            return row
    raise KeyError(comparison)


def main() -> None:
    _write_training_table()
    training = {row["comparison"]: row for row in csv.DictReader(TRAIN.open(encoding="utf-8"))}
    panels = [
        (_guided("rl_greedy_minus_bc_greedy"), "RL greedy\nminus BC greedy"),
        (_guided("bc_wide_minus_bc_greedy"), "BC guided\nminus BC greedy"),
        (_guided("rl_wide_minus_bc_wide"), "RL guided\nminus BC guided"),
        (training["imitation_minus_bc_greedy"], "Imitation\nminus BC greedy"),
        (training["rl_imitation_minus_imitation"], "RL+imitation\nminus imitation"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    for index, (row, _label) in enumerate(panels):
        mean = float(row["mean_h"])
        low = float(row["ci_low_h"])
        high = float(row["ci_high_h"])
        color = "#1f4e79" if index < 3 else "#8d6a4a"
        ax.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="o", color=color, capsize=4, markersize=6)
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_xticks(range(len(panels)))
    ax.set_xticklabels([label for _row, label in panels])
    ax.set_ylabel("Paired duration difference (h)")
    ax.set_title("Coverable development instances, five seeds")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
