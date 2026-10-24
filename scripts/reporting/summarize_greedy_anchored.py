"""Summarize the frozen greedy-plus-three validation comparison."""

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

RAW = ROOT / "results" / "raw" / "bc_rl_improvement_v1" / "teacher_validation" / "evaluations.csv"
OUT = ROOT / "results" / "tables" / "final"
T_CRIT = float(stats.t.ppf(0.975, 4))


def _rows():
    with RAW.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        "oracle_solver_executions": sum(int(float(row["oracle_solver_executions"])) for row in ordered),
        "n_candidates": len(ordered),
    }


def _mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _penalized(value, feasible, cap=100.0):
    return value if feasible and math.isfinite(value) else cap


def _paired(bc_by_seed, rl_by_seed):
    seeds = sorted(set(bc_by_seed) & set(rl_by_seed))
    deltas = [rl_by_seed[seed] - bc_by_seed[seed] for seed in seeds]
    array = np.asarray(deltas, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = T_CRIT * sd / math.sqrt(len(array))
    t_result = stats.ttest_1samp(array, 0.0)
    nonzero = [value for value in deltas if value != 0.0]
    sign = stats.binomtest(sum(value < 0 for value in nonzero), len(nonzero), 0.5, alternative="two-sided")
    wilcoxon = stats.wilcoxon(array, alternative="two-sided", method="exact")
    return {
        "seeds": "|".join(str(seed) for seed in seeds),
        "deltas_h": "|".join(f"{value:.6f}" for value in deltas),
        "n_negative": sum(value < 0 for value in deltas),
        "mean_h": mean,
        "sd_h": sd,
        "ci_low_h": mean - half,
        "ci_high_h": mean + half,
        "t_p": float(t_result.pvalue),
        "sign_p": float(sign.pvalue),
        "wilcoxon_p": float(wilcoxon.pvalue),
    }


def main() -> None:
    rows = _rows()
    expected = 5 * 2 * 2 * 100 * 4
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} construction rows, found {len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        key = (row["temperature"], row["seed"], row["role"], row["instance_index"])
        groups[key].append(row)
    anchored_rows = []
    for key, group in groups.items():
        if len(group) != 4:
            raise SystemExit(f"incomplete group {key}")
        summary = _anchored(group)
        temperature, seed, role, index = key
        anchored_rows.append(
            {
                "temperature": temperature,
                "seed": seed,
                "role": role,
                "instance_index": index,
                **summary,
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    seed_path = OUT / "greedy_anchored_by_seed.csv"
    diff_path = OUT / "greedy_anchored_paired.csv"
    seed_rows = []
    for temperature in ("1.0", "0.8"):
        for role in ("bc", "rl"):
            for seed in ("42", "43", "44", "45", "46"):
                selected = [
                    row for row in anchored_rows
                    if row["temperature"] == temperature and row["role"] == role and row["seed"] == seed
                ]
                greedy = [
                    row for row in rows
                    if row["temperature"] == temperature and row["role"] == role and row["seed"] == seed and row["protocol"] == "greedy"
                ]
                samples = [
                    row for row in rows
                    if row["temperature"] == temperature and row["role"] == role and row["seed"] == seed and row["protocol"] == "sample" and row["feasible"] == "1"
                ]
                seed_rows.append(
                    {
                        "temperature": temperature,
                        "role": role,
                        "seed": seed,
                        "anchored_h": _mean([row["objective_h"] for row in selected]),
                        "anchored_penalized_h": _mean(
                            [_penalized(row["objective_h"], row["feasible"]) for row in selected]
                        ),
                        "anchored_travel_h": _mean([row["travel_h"] for row in selected]),
                        "anchored_charging_h": _mean([row["charging_h"] for row in selected]),
                        "anchored_feasible_rate": _mean([row["feasible"] for row in selected]),
                        "anchored_runtime_s": _mean([row["runtime_s"] for row in selected]),
                        "anchored_executions": _mean([row["oracle_solver_executions"] for row in selected]),
                        "greedy_h": _mean([float(row["objective_h"]) for row in greedy if row["feasible"] == "1"]),
                        "greedy_runtime_s": _mean([float(row["runtime_s"]) for row in greedy]),
                        "greedy_executions": _mean([int(float(row["oracle_solver_executions"])) for row in greedy]),
                        "sample_mean_h": _mean([float(row["objective_h"]) for row in samples]),
                    }
                )
    with seed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    paired = []
    for temperature in ("1.0", "0.8"):
        for metric in ("anchored_h", "anchored_penalized_h", "greedy_h", "sample_mean_h"):
            bc = {
                row["seed"]: float(row[metric])
                for row in seed_rows
                if row["temperature"] == temperature and row["role"] == "bc"
            }
            rl = {
                row["seed"]: float(row[metric])
                for row in seed_rows
                if row["temperature"] == temperature and row["role"] == "rl"
            }
            stats_row = _paired(bc, rl)
            bc_mean = sum(bc.values()) / len(bc)
            stats_row.update(
                {
                    "temperature": temperature,
                    "metric": metric,
                    "bc_h": bc_mean,
                    "rl_h": sum(rl.values()) / len(rl),
                    "percent": 100.0 * stats_row["mean_h"] / bc_mean,
                }
            )
            paired.append(stats_row)
    with diff_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    for row in paired:
        print(
            row["temperature"],
            row["metric"],
            f"{row['mean_h']:.4f}",
            f"[{row['ci_low_h']:.4f}, {row['ci_high_h']:.4f}]",
            f"{row['n_negative']}/5",
            f"sign {row['sign_p']:.4f}",
        )


if __name__ == "__main__":
    main()
