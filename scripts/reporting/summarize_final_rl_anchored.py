"""Paired summary of greedy-anchored decoding on the greedy-rollout checkpoints."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "results" / "raw" / "bc_rl_final_v1" / "anchored_validation" / "evaluations.csv"
OUT = ROOT / "results" / "tables" / "final" / "greedy_rollout_anchored_paired.csv"
T_CRIT = float(stats.t.ppf(0.975, 4))
CAP = 100.0


def _anchored(group):
    ordered = sorted(group, key=lambda row: (0 if row["protocol"] == "greedy" else 1, int(row["trial"])))
    feasible = [row for row in ordered if row["feasible"] == "1" and row["objective_h"] != ""]
    chosen = min(feasible, key=lambda row: float(row["objective_h"])) if feasible else None
    return {
        "feasible": int(chosen is not None),
        "objective_h": float(chosen["objective_h"]) if chosen else math.nan,
        "travel_h": float(chosen["travel_h"]) if chosen else math.nan,
        "charging_h": float(chosen["charging_h"]) if chosen else math.nan,
        "runtime_s": sum(float(row["runtime_s"]) for row in ordered),
        "executions": sum(int(float(row["oracle_solver_executions"])) for row in ordered),
    }


def _mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _paired(deltas):
    array = np.asarray(deltas, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = T_CRIT * sd / math.sqrt(len(array))
    t_result = stats.ttest_1samp(array, 0.0)
    nonzero = [value for value in deltas if value != 0.0]
    sign = stats.binomtest(
        sum(value < 0 for value in nonzero), len(nonzero), 0.5, alternative="two-sided"
    )
    wilcoxon = stats.wilcoxon(array, alternative="two-sided", method="exact")
    return {
        "n_negative": sum(value < 0 for value in deltas),
        "mean_h": mean,
        "sd_h": sd,
        "ci_low_h": mean - half,
        "ci_high_h": mean + half,
        "t_p": float(t_result.pvalue),
        "sign_p": float(sign.pvalue),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "deltas_h": "|".join(f"{value:.6f}" for value in deltas),
    }


def main() -> None:
    with RAW.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2 * 5 * 2 * 100 * 4:
        raise SystemExit(f"expected 8000 rows, found {len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["seed"], row["role"], row["instance_index"])].append(row)
    records = []
    for key, group in groups.items():
        model, seed, role, _index = key
        summary = _anchored(group)
        greedy = next(row for row in group if row["protocol"] == "greedy")
        samples = [row for row in group if row["protocol"] == "sample" and row["feasible"] == "1"]
        records.append(
            {
                "model": model,
                "seed": seed,
                "role": role,
                "anchored_h": summary["objective_h"] if summary["feasible"] else CAP,
                "anchored_feasible": summary["feasible"],
                "travel_h": summary["travel_h"],
                "charging_h": summary["charging_h"],
                "runtime_s": summary["runtime_s"],
                "executions": summary["executions"],
                "greedy_h": float(greedy["objective_h"]) if greedy["feasible"] == "1" else CAP,
                "sample_mean_h": _mean([float(row["objective_h"]) for row in samples]),
                "sample_travel_h": _mean([float(row["travel_h"]) for row in samples]),
                "sample_charging_h": _mean([float(row["charging_h"]) for row in samples]),
            }
        )
    seed_rows = []
    for model in ("H2", "H4"):
        for role in ("bc", "rl"):
            for seed in ("42", "43", "44", "45", "46"):
                selected = [
                    row for row in records
                    if row["model"] == model and row["role"] == role and row["seed"] == seed
                ]
                seed_rows.append(
                    {
                        "model": model,
                        "role": role,
                        "seed": seed,
                        "anchored_h": _mean([row["anchored_h"] for row in selected]),
                        "greedy_h": _mean([row["greedy_h"] for row in selected]),
                        "sample_mean_h": _mean([row["sample_mean_h"] for row in selected]),
                        "travel_h": _mean([row["travel_h"] for row in selected]),
                        "charging_h": _mean([row["charging_h"] for row in selected]),
                        "sample_travel_h": _mean([row["sample_travel_h"] for row in selected]),
                        "sample_charging_h": _mean([row["sample_charging_h"] for row in selected]),
                        "runtime_s": _mean([row["runtime_s"] for row in selected]),
                        "executions": _mean([row["executions"] for row in selected]),
                        "anchored_feasible": _mean([row["anchored_feasible"] for row in selected]),
                    }
                )
    paired_rows = []
    for model in ("H2", "H4"):
        for metric in ("anchored_h", "greedy_h", "sample_mean_h", "travel_h", "charging_h", "sample_travel_h", "sample_charging_h"):
            bc = {row["seed"]: row[metric] for row in seed_rows if row["model"] == model and row["role"] == "bc"}
            rl = {row["seed"]: row[metric] for row in seed_rows if row["model"] == model and row["role"] == "rl"}
            seeds = ["42", "43", "44", "45", "46"]
            deltas = [rl[seed] - bc[seed] for seed in seeds]
            stats_row = _paired(deltas)
            bc_mean = sum(bc[seed] for seed in seeds) / 5
            rl_mean = sum(rl[seed] for seed in seeds) / 5
            runtime = _mean([row["runtime_s"] for row in seed_rows if row["model"] == model])
            executions = _mean([row["executions"] for row in seed_rows if row["model"] == model])
            feasible = _mean([row["anchored_feasible"] for row in seed_rows if row["model"] == model])
            stats_row.update(
                {
                    "model": model,
                    "metric": metric,
                    "bc_h": bc_mean,
                    "rl_h": rl_mean,
                    "percent": 100.0 * stats_row["mean_h"] / bc_mean if bc_mean else math.nan,
                    "mean_anchored_runtime_s": runtime,
                    "mean_anchored_executions": executions,
                    "anchored_feasible_rate": feasible,
                }
            )
            paired_rows.append(stats_row)
            print(
                model,
                metric,
                f"{stats_row['mean_h']:.4f}",
                f"[{stats_row['ci_low_h']:.4f}, {stats_row['ci_high_h']:.4f}]",
                f"{stats_row['n_negative']}/5",
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)


if __name__ == "__main__":
    main()
