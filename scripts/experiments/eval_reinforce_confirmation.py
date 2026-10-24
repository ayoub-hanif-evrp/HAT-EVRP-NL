"""Score the frozen BC, leave-one-out, and return-to-go checkpoints once.

The confirmation seeds were named in the protocol before training. This script
does not select a checkpoint.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import FixedRouteOracle, decompose_solution
from data import generate_synthetic
from train import decode, load_policy

OUT = ROOT / "results" / "raw" / "reinforce_diagnosis_v1" / "confirmation"
RUNS = ROOT / "results" / "raw" / "reinforce_diagnosis_v1" / "runs"
PROTOCOL = json.loads((ROOT / "results" / "raw" / "reinforce_diagnosis_v1" / "frozen_protocol.json").read_text(encoding="utf-8"))


def _hours(result, instance) -> float:
    if result["feasible"]:
        return float(result["objective"])
    return float(len(instance.customers) * instance.max_route_hours)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instances = []
    rows_meta = []
    for offset in range(PROTOCOL["n_confirm"]):
        seed = PROTOCOL["confirm_seed_start"] + offset
        instance = generate_synthetic(PROTOCOL["size"], seed, generator_version="v2")
        oracle = FixedRouteOracle(instance)
        coverable = int(all(oracle.solve((customer,)).feasible for customer in instance.customers))
        instances.append(instance)
        rows_meta.append({"instance_index": offset, "seed": seed, "name": instance.name, "physically_coverable": coverable})
    with (OUT / "instances.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_meta[0]))
        writer.writeheader()
        writer.writerows(rows_meta)
    fields = [
        "candidate", "seed", "instance_index", "physically_coverable", "protocol",
        "feasible", "objective_h", "penalized_h", "travel_h", "service_h", "charging_h",
        "runtime_s", "oracle_solver_executions",
    ]
    out_path = OUT / "evaluations.csv"
    done = set()
    if out_path.exists():
        done = {
            (row["candidate"], row["seed"], row["instance_index"], row["protocol"])
            for row in csv.DictReader(out_path.open(encoding="utf-8"))
        }
    candidates = []
    for seed in PROTOCOL["seeds"]:
        candidates.append(("bc", seed, RUNS / f"H2_seed{seed}_loo" / "bc_best.pt"))
        candidates.append(("loo", seed, RUNS / f"H2_seed{seed}_loo" / "rl_best.pt"))
        candidates.append(("rtg", seed, RUNS / f"H2_seed{seed}_rtg" / "rl_best.pt"))
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not done:
            writer.writeheader()
        for name, seed, path in candidates:
            policy = load_policy(path)
            for index, instance in enumerate(instances):
                jobs = [("greedy", False, 0)] + [("sample", True, trial) for trial in range(3)]
                for protocol, sample, trial in jobs:
                    key = (name, str(seed), str(index), protocol if protocol == "greedy" else f"sample{trial}")
                    label = "greedy" if protocol == "greedy" else f"sample{trial}"
                    if (name, str(seed), str(index), label) in done:
                        continue
                    with torch.no_grad():
                        result = decode(
                            policy,
                            instance,
                            sample=sample,
                            seed=0 if not sample else 982000 + index * 16 + trial,
                        )
                    parts = decompose_solution(instance, result["plans"]) if result["feasible"] else {}
                    writer.writerow(
                        {
                            "candidate": name,
                            "seed": seed,
                            "instance_index": index,
                            "physically_coverable": rows_meta[index]["physically_coverable"],
                            "protocol": label,
                            "feasible": int(bool(result["feasible"])),
                            "objective_h": "" if not result["feasible"] else result["objective"],
                            "penalized_h": _hours(result, instance),
                            "travel_h": parts.get("travel_h", ""),
                            "service_h": parts.get("service_h", ""),
                            "charging_h": parts.get("charging_h", ""),
                            "runtime_s": result["runtime"],
                            "oracle_solver_executions": result["oracle_solver_executions"],
                        }
                    )
                    handle.flush()
            print({"scored": name, "seed": seed}, flush=True)


if __name__ == "__main__":
    main()
