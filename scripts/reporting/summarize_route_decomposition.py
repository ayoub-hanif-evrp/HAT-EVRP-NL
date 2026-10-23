"""Paired BC versus RL duration components from the regenerated routes."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "raw" / "bc_rl_route_decomposition_v1" / "solutions.csv"
OUT = ROOT / "results" / "tables" / "final" / "route_decomposition_summary.csv"
T_CRIT = 2.7764451051977987
FIELDS = ("travel_h", "service_h", "charging_h", "full_h", "n_charges", "energy_charged_wh", "oracle_calls")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _ci(deltas: list[float]) -> tuple[float, float, float]:
    mean = _mean(deltas)
    var = sum((delta - mean) ** 2 for delta in deltas) / (len(deltas) - 1)
    half = T_CRIT * math.sqrt(var / len(deltas))
    return mean, mean - half, mean + half


def main() -> None:
    with SRC.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4000:
        raise SystemExit(f"decomposition has {len(rows)} rows, expected 4000")
    if any(abs(float(row["objective_abs_error_h"])) > 1e-4 for row in rows):
        raise SystemExit("a regenerated objective disagrees with the locked evaluation")
    if any(row["identity_ok"] != "1" or row["feasible"] != "1" for row in rows):
        raise SystemExit("a regenerated route failed the duration identity or feasibility check")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["seed"], row["role"], row["size"], row["instance"], row["protocol"])].append(row)
    summary = []
    for protocol in ("greedy", "sample"):
        for model in ("H2", "H4"):
            for size in ("10", "20"):
                for field in FIELDS:
                    seed_deltas = []
                    seed_bc = []
                    seed_rl = []
                    changed = []
                    for seed in ("42", "43", "44", "45", "46"):
                        instances = {
                            key[4]
                            for key in grouped
                            if key[0] == model and key[1] == seed and key[3] == size and key[5] == protocol
                        }
                        if len(instances) != 20:
                            raise SystemExit(f"{model} seed {seed} n={size} {protocol} has {len(instances)} instances")
                        deltas = []
                        bc_means = []
                        rl_means = []
                        n_changed = 0
                        n_pairs = 0
                        for instance in instances:
                            bc = grouped[(model, seed, "bc", size, instance, protocol)]
                            rl = grouped[(model, seed, "rl", size, instance, protocol)]
                            expected = 1 if protocol == "greedy" else 4
                            if len(bc) != expected or len(rl) != expected:
                                raise SystemExit(f"trial count mismatch for {model} {seed} {instance} {protocol}")
                            bc_value = _mean([float(row[field]) for row in bc])
                            rl_value = _mean([float(row[field]) for row in rl])
                            deltas.append(rl_value - bc_value)
                            bc_means.append(bc_value)
                            rl_means.append(rl_value)
                            for left, right in zip(bc, rl):
                                n_pairs += 1
                                if left["fingerprint"] != right["fingerprint"]:
                                    n_changed += 1
                        seed_deltas.append(_mean(deltas))
                        seed_bc.append(_mean(bc_means))
                        seed_rl.append(_mean(rl_means))
                        changed.append(n_changed / n_pairs)
                    mean, low, high = _ci(seed_deltas)
                    summary.append(
                        {
                            "model": model,
                            "size": size,
                            "protocol": protocol,
                            "component": field,
                            "bc_mean": _mean(seed_bc),
                            "rl_mean": _mean(seed_rl),
                            "delta": mean,
                            "ci_low": low,
                            "ci_high": high,
                            "seeds_lower": sum(delta < -1e-8 for delta in seed_deltas),
                            "fingerprint_change_rate": _mean(changed) if field == "full_h" else "",
                        }
                    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        if row["component"] in {"travel_h", "service_h", "charging_h", "full_h"} and row["protocol"] == "sample":
            print(
                row["model"],
                row["size"],
                row["component"],
                f"{row['delta']:.4f}",
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
                row["seeds_lower"],
            )


if __name__ == "__main__":
    main()
