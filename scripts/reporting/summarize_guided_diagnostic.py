"""Paired development summary for the frozen guided-decoder diagnostic."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw" / "algorithmic_superiority_v1" / "guided_diagnostic" / "evaluations.csv"
DIV = ROOT / "results" / "raw" / "algorithmic_superiority_v1" / "guided_diagnostic" / "divergences.csv"
DEV = ROOT / "results" / "raw" / "algorithmic_superiority_v1" / "guided_diagnostic" / "deviations.csv"
OUT = ROOT / "results" / "tables" / "final" / "guided_decoder_paired.csv"
T_CRIT = float(stats.t.ppf(0.975, 4))


def _paired(deltas: list[float]) -> dict:
    array = np.asarray(deltas, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = T_CRIT * sd / math.sqrt(len(array))
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
    return {
        "mean_h": mean,
        "ci_low_h": mean - half,
        "ci_high_h": mean + half,
        "n_negative": int(sum(value < 0 for value in deltas)),
        "t_p": float(stats.ttest_1samp(array, 0.0).pvalue),
        "sign_p": sign_p,
        "wilcoxon_p": wilcoxon_p,
        "deltas_h": "|".join(f"{value:.6f}" for value in deltas),
    }


def _mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    rows = list(csv.DictReader(RAW.open(encoding="utf-8")))
    by_seed_policy = {}
    for row in rows:
        by_seed_policy.setdefault((row["seed"], row["policy"]), []).append(row)
    seeds = sorted({row["seed"] for row in rows})
    summary = []

    def add(comparison, subset, left_key, right_key, left_policy, right_policy):
        deltas = []
        relatives = []
        left_means = []
        right_means = []
        for seed in seeds:
            left_rows = by_seed_policy[(seed, left_policy)]
            right_rows = by_seed_policy[(seed, right_policy)]
            if subset == "coverable":
                left_rows = [row for row in left_rows if row["physically_coverable"] == "1"]
                right_rows = [row for row in right_rows if row["physically_coverable"] == "1"]
            left_mean = _mean(left_rows, left_key)
            right_mean = _mean(right_rows, right_key)
            deltas.append(right_mean - left_mean)
            relatives.append((right_mean - left_mean) / left_mean)
            left_means.append(left_mean)
            right_means.append(right_mean)
        stats_row = _paired(deltas)
        stats_row.update(
            {
                "comparison": comparison,
                "subset": subset,
                "n_instances": len([row for row in by_seed_policy[(seeds[0], left_policy)] if subset != "coverable" or row["physically_coverable"] == "1"]),
                "left_h": sum(left_means) / len(left_means),
                "right_h": sum(right_means) / len(right_means),
                "mean_relative": sum(relatives) / len(relatives),
                "relative_deltas": "|".join(f"{value:.6f}" for value in relatives),
            }
        )
        summary.append(stats_row)
        print(
            comparison,
            subset,
            f"{stats_row['mean_h']:.4f}",
            f"[{stats_row['ci_low_h']:.4f}, {stats_row['ci_high_h']:.4f}]",
            f"{stats_row['n_negative']}/5",
            f"rel {stats_row['mean_relative']*100:.2f}%",
        )

    # Per-instance best of BC-wide and RL-wide versus BC greedy. Same instances, paired by index.
    for subset in ("all", "coverable"):
        deltas = []
        relatives = []
        left_means = []
        right_means = []
        for seed in seeds:
            bc = by_seed_policy[(seed, "bc")]
            rl = {row["instance_index"]: row for row in by_seed_policy[(seed, "rl")]}
            chosen = []
            for row in bc:
                if subset == "coverable" and row["physically_coverable"] != "1":
                    continue
                other = rl[row["instance_index"]]
                best = min(float(row["wide_h"]), float(other["wide_h"]))
                chosen.append((float(row["greedy_h"]), best))
            left = sum(item[0] for item in chosen) / len(chosen)
            right = sum(item[1] for item in chosen) / len(chosen)
            deltas.append(right - left)
            relatives.append((right - left) / left)
            left_means.append(left)
            right_means.append(right)
        stats_row = _paired(deltas)
        stats_row.update(
            {
                "comparison": "best_wide_minus_bc_greedy",
                "subset": subset,
                "n_instances": len(chosen),
                "left_h": sum(left_means) / len(left_means),
                "right_h": sum(right_means) / len(right_means),
                "mean_relative": sum(relatives) / len(relatives),
                "relative_deltas": "|".join(f"{value:.6f}" for value in relatives),
            }
        )
        summary.append(stats_row)
        print(
            "best_wide_minus_bc_greedy",
            subset,
            f"{stats_row['mean_h']:.4f}",
            f"[{stats_row['ci_low_h']:.4f}, {stats_row['ci_high_h']:.4f}]",
            f"{stats_row['n_negative']}/5",
            f"rel {stats_row['mean_relative']*100:.2f}%",
            "n",
            len(chosen),
        )

    pairs = [
        ("rl_greedy_minus_bc_greedy", "greedy_h", "greedy_h", "bc", "rl"),
        ("bc_narrow_minus_bc_greedy", "greedy_h", "narrow_h", "bc", "bc"),
        ("bc_wide_minus_bc_greedy", "greedy_h", "wide_h", "bc", "bc"),
        ("rl_narrow_minus_rl_greedy", "greedy_h", "narrow_h", "rl", "rl"),
        ("rl_wide_minus_rl_greedy", "greedy_h", "wide_h", "rl", "rl"),
        ("rl_wide_minus_bc_wide", "wide_h", "wide_h", "bc", "rl"),
        ("rl_greedy_minus_bc_wide", "wide_h", "greedy_h", "bc", "rl"),
        ("lambda4_minus_greedy_bc", "greedy_h", "lambda4_h", "bc", "bc"),
        ("lambda4_minus_greedy_rl", "greedy_h", "lambda4_h", "rl", "rl"),
    ]
    for name, left_key, right_key, left_policy, right_policy in pairs:
        for subset in ("all", "coverable"):
            add(name, subset, left_key, right_key, left_policy, right_policy)

    if DIV.exists():
        div_rows = list(csv.DictReader(DIV.open(encoding="utf-8")))
        same = sum(row["same_fingerprint"] == "1" for row in div_rows)
        print("same_greedy_fingerprint", same, "/", len(div_rows))
    if DEV.exists():
        dev_rows = [row for row in csv.DictReader(DEV.open(encoding="utf-8")) if row["budget"] == "wide" and row["improved"] == "1"]
        print("wide_improving_deviations", len(dev_rows))
    gate = next(row for row in summary if row["comparison"] == "best_wide_minus_bc_greedy" and row["subset"] == "coverable")
    passed = gate["mean_relative"] <= -0.01 and gate["n_negative"] >= 4 and gate["ci_high_h"] < 0
    print("TRAINING_GATE", "PASS" if passed else "STOP")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
