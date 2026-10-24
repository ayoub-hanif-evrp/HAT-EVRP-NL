"""Separate physically unsolvable instances from the unbiased comparison.

A customer with no feasible dedicated route cannot be covered by a longer route
under Euclidean travel. Instances that fail this test are reported separately.
The duration comparison is then restricted to instances both constructors solve.
"""

from __future__ import annotations

import csv
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import FixedRouteOracle
from solver import randomized_greedy

RAW = ROOT / "results" / "raw" / "final_autonomous_study" / "unbiased_eval" / "evaluations.csv"
INSTANCES = ROOT / "results" / "raw" / "final_autonomous_study" / "splits" / "instances.pkl"
OUT = ROOT / "results" / "tables" / "final" / "solvable_subset_paired.csv"
AUDIT = ROOT / "results" / "tables" / "final" / "singleton_feasibility_audit.csv"
T_CRIT = float(stats.t.ppf(0.975, 4))
SPLITS = ("dev_v2", "confirm_v2", "shift_v2c", "shift_slow", "shift_geo")


def _chosen(group):
    ordered = sorted(group, key=lambda row: (0 if row["protocol"] == "greedy" else 1, int(row["trial"])))
    feasible = [row for row in ordered if row["feasible"] == "1" and row["objective_h"]]
    return min(feasible, key=lambda row: float(row["objective_h"])) if feasible else None


def _paired(deltas):
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
    t_p = float(stats.ttest_1samp(array, 0.0).pvalue)
    return mean, mean - half, mean + half, int(sum(value < 0 for value in deltas)), t_p, sign_p, wilcoxon_p


def main() -> None:
    blob = pickle.loads(INSTANCES.read_bytes())
    rows = list(csv.DictReader(RAW.open(encoding="utf-8")))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["candidate"], row["seed"], row["instance_index"])].append(row)
    audit_rows = []
    for split in ("dev_v2", "confirm_v2"):
        instances = blob["splits"][split]
        for index, instance in enumerate(instances):
            oracle = FixedRouteOracle(instance)
            singletons = sum(1 for customer in instance.customers if oracle.solve((customer,)).feasible)
            neural = _chosen(groups[(split, "original_bc", "42", str(index))]) is not None
            strong = None
            if singletons < len(instance.customers):
                result = randomized_greedy(
                    instance,
                    restarts=8,
                    seed=7000 + index,
                    local_search_passes=3,
                    local_search_budget=2500,
                    local_operators="strong",
                )
                strong = int(bool(result.feasible))
            audit_rows.append(
                {
                    "split": split,
                    "instance_index": index,
                    "instance": instance.name,
                    "feasible_singletons": singletons,
                    "n_customers": len(instance.customers),
                    "physically_coverable": int(singletons == len(instance.customers)),
                    "neural_bc_seed42_feasible": int(neural),
                    "strong_search_feasible": "" if strong is None else strong,
                }
            )
    paired_rows = []
    for split in SPLITS:
        for candidate in ("original_rl", "greedy_baseline_rl"):
            deltas = []
            bc_means = []
            rl_means = []
            n_instances = None
            for seed in ("42", "43", "44", "45", "46"):
                indices = sorted({key[3] for key in groups if key[0] == split and key[2] == seed})
                bc_values = []
                rl_values = []
                for index in indices:
                    bc = _chosen(groups[(split, "original_bc", seed, index)])
                    rl = _chosen(groups[(split, candidate, seed, index)])
                    if bc is None or rl is None:
                        continue
                    bc_values.append(float(bc["objective_h"]))
                    rl_values.append(float(rl["objective_h"]))
                n_instances = len(bc_values)
                deltas.append(sum(rl_values) / len(rl_values) - sum(bc_values) / len(bc_values))
                bc_means.append(sum(bc_values) / len(bc_values))
                rl_means.append(sum(rl_values) / len(rl_values))
            mean, low, high, n_neg, t_p, sign_p, wilcoxon_p = _paired(deltas)
            paired_rows.append(
                {
                    "split": split,
                    "candidate": candidate,
                    "n_instances": n_instances,
                    "bc_h": sum(bc_means) / len(bc_means),
                    "rl_h": sum(rl_means) / len(rl_means),
                    "mean_h": mean,
                    "ci_low_h": low,
                    "ci_high_h": high,
                    "n_negative": n_neg,
                    "t_p": t_p,
                    "sign_p": sign_p,
                    "wilcoxon_p": wilcoxon_p,
                    "deltas_h": "|".join(f"{value:.6f}" for value in deltas),
                }
            )
            print(split, candidate, f"{mean:.4f}", f"[{low:.4f}, {high:.4f}]", f"{n_neg}/5", "n", n_instances)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)


if __name__ == "__main__":
    main()
