"""Paired penalized summaries for an unbiased-evaluation CSV."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "raw" / "final_autonomous_study" / "unbiased_eval" / "evaluations.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "tables" / "final" / "unbiased_eval_paired.csv"
T_CRIT = float(stats.t.ppf(0.975, 4))
CAP = 100.0


def _penalized(row) -> float:
    if row["feasible"] == "1" and row["objective_h"] != "":
        return float(row["objective_h"])
    return CAP


def _group_metrics(rows):
    ordered = sorted(rows, key=lambda row: (0 if row["protocol"] == "greedy" else 1, int(row["trial"])))
    feasible = [row for row in ordered if row["feasible"] == "1" and row["objective_h"] != ""]
    chosen = min(feasible, key=lambda row: float(row["objective_h"])) if feasible else None
    samples = [row for row in ordered if row["protocol"] == "sample"]
    greedy = next(row for row in ordered if row["protocol"] == "greedy")
    return {
        "anchored": float(chosen["objective_h"]) if chosen else CAP,
        "anchored_feasible": int(chosen is not None),
        "anchored_travel": float(chosen["travel_h"]) if chosen else math.nan,
        "anchored_charging": float(chosen["charging_h"]) if chosen else math.nan,
        "greedy": _penalized(greedy),
        "sample_mean": sum(_penalized(row) for row in samples) / len(samples),
        "runtime_s": sum(float(row["runtime_s"]) for row in ordered),
        "executions": sum(int(float(row["oracle_solver_executions"])) for row in ordered),
    }


def _paired(deltas):
    array = np.asarray(deltas, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    half = T_CRIT * sd / math.sqrt(len(array))
    t_result = stats.ttest_1samp(array, 0.0)
    nonzero = [value for value in deltas if value != 0.0]
    if nonzero:
        sign = stats.binomtest(
            sum(value < 0 for value in nonzero), len(nonzero), 0.5, alternative="two-sided"
        )
        sign_p = float(sign.pvalue)
    else:
        sign_p = 1.0
    if np.allclose(array, 0.0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(array, alternative="two-sided", method="exact").pvalue)
    return {
        "n_negative": int(sum(value < 0 for value in deltas)),
        "mean_h": mean,
        "sd_h": sd,
        "ci_low_h": mean - half,
        "ci_high_h": mean + half,
        "t_p": float(t_result.pvalue),
        "sign_p": sign_p,
        "wilcoxon_p": wilcoxon_p,
        "deltas_h": "|".join(f"{value:.6f}" for value in deltas),
    }


def main() -> None:
    with RAW.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["candidate"], row["seed"], row["instance_index"])].append(row)
    instance_rows = []
    for key, group in groups.items():
        split, candidate, seed, _index = key
        metrics = _group_metrics(group)
        instance_rows.append({"split": split, "candidate": candidate, "seed": seed, **metrics})
    seed_rows = []
    for split in sorted({row["split"] for row in instance_rows}):
        for candidate in sorted({row["candidate"] for row in instance_rows if row["split"] == split}):
            for seed in ("42", "43", "44", "45", "46"):
                selected = [
                    row for row in instance_rows
                    if row["split"] == split and row["candidate"] == candidate and row["seed"] == seed
                ]
                if not selected:
                    continue
                seed_rows.append(
                    {
                        "split": split,
                        "candidate": candidate,
                        "seed": seed,
                        "n": len(selected),
                        "anchored_h": sum(row["anchored"] for row in selected) / len(selected),
                        "greedy_h": sum(row["greedy"] for row in selected) / len(selected),
                        "sample_mean_h": sum(row["sample_mean"] for row in selected) / len(selected),
                        "feasible_rate": sum(row["anchored_feasible"] for row in selected) / len(selected),
                        "travel_h": float(np.nanmean([row["anchored_travel"] for row in selected])),
                        "charging_h": float(np.nanmean([row["anchored_charging"] for row in selected])),
                        "runtime_s": sum(row["runtime_s"] for row in selected) / len(selected),
                        "executions": sum(row["executions"] for row in selected) / len(selected),
                    }
                )
    comparisons = []
    reference = "original_bc"
    for split in sorted({row["split"] for row in seed_rows}):
        for candidate in sorted({row["candidate"] for row in seed_rows if row["split"] == split}):
            if candidate == reference:
                continue
            for metric in ("anchored_h", "greedy_h", "sample_mean_h", "travel_h", "charging_h"):
                bc = {
                    row["seed"]: row[metric]
                    for row in seed_rows
                    if row["split"] == split and row["candidate"] == reference
                }
                other = {
                    row["seed"]: row[metric]
                    for row in seed_rows
                    if row["split"] == split and row["candidate"] == candidate
                }
                seeds = [seed for seed in ("42", "43", "44", "45", "46") if seed in bc and seed in other]
                deltas = [other[seed] - bc[seed] for seed in seeds]
                stats_row = _paired(deltas)
                bc_mean = sum(bc[seed] for seed in seeds) / len(seeds)
                other_mean = sum(other[seed] for seed in seeds) / len(seeds)
                feas = [
                    row["feasible_rate"]
                    for row in seed_rows
                    if row["split"] == split and row["candidate"] == candidate
                ]
                stats_row.update(
                    {
                        "split": split,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "bc_h": bc_mean,
                        "candidate_h": other_mean,
                        "feasible_rate": sum(feas) / len(feas),
                    }
                )
                comparisons.append(stats_row)
                if metric == "anchored_h":
                    print(
                        split,
                        candidate,
                        f"{stats_row['mean_h']:.4f}",
                        f"[{stats_row['ci_low_h']:.4f}, {stats_row['ci_high_h']:.4f}]",
                        f"{stats_row['n_negative']}/5",
                        f"feas {stats_row['feasible_rate']:.3f}",
                    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    seed_path = OUT.with_name("unbiased_eval_by_seed.csv")
    with seed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)


if __name__ == "__main__":
    main()
