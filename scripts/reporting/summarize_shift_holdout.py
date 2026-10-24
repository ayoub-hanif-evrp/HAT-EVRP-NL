"""Summarize the frozen slow-charger holdout. Does not reselect checkpoints."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "raw" / "shift_holdout_v1" / "evaluations.csv"
OUT = ROOT / "results" / "tables" / "final" / "shift_holdout_summary.csv"
T_CRIT = 2.7764451051977987


def main() -> None:
    with SRC.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1000:
        raise SystemExit(f"expected 1000 holdout rows, found {len(rows)}")
    if any(row["feasible"] != "1" for row in rows):
        raise SystemExit("a holdout decode is infeasible")
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["role"], row["instance_index"], row["protocol"])].append(float(row["objective_h"]))
    summary = []
    for protocol in ("greedy", "sample"):
        deltas = []
        bc_means = []
        rl_means = []
        for seed in ("42", "43", "44", "45", "46"):
            bc = []
            rl = []
            for index in range(20):
                left = grouped[(seed, "bc", str(index), protocol)]
                right = grouped[(seed, "rl", str(index), protocol)]
                if protocol == "greedy":
                    bc.append(left[0])
                    rl.append(right[0])
                else:
                    if len(left) != 4 or len(right) != 4:
                        raise SystemExit("sample count is not four")
                    bc.append(sum(left) / 4)
                    rl.append(sum(right) / 4)
            bc_means.append(sum(bc) / 20)
            rl_means.append(sum(rl) / 20)
            deltas.append(rl_means[-1] - bc_means[-1])
        array = np.asarray(deltas)
        mean = float(array.mean())
        sd = float(array.std(ddof=1))
        half = T_CRIT * sd / np.sqrt(5)
        t_p = float(stats.ttest_1samp(array, 0.0).pvalue)
        sign_p = float(stats.binomtest(int(np.sum(array > 0)), 5, 0.5).pvalue)
        summary.append(
            {
                "protocol": "greedy" if protocol == "greedy" else "sampled_mean",
                "bc_h": float(np.mean(bc_means)),
                "rl_h": float(np.mean(rl_means)),
                "delta_h": mean,
                "percent": 100.0 * mean / float(np.mean(bc_means)),
                "sd_h": sd,
                "ci_low_h": mean - half,
                "ci_high_h": mean + half,
                "t_p": t_p,
                "sign_p": sign_p,
                "seeds_rl_lower": int(np.sum(array < 0)),
                "seed_deltas_h": "|".join(f"{delta:.6f}" for delta in deltas),
                "feasible_rate": 1.0,
                "status": "frozen holdout, not used to reselect checkpoints",
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row["protocol"], f"{row['delta_h']:.4f}", row["ci_low_h"], row["ci_high_h"], row["seeds_rl_lower"], row["sign_p"])


if __name__ == "__main__":
    main()
