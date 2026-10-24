"""Greedy-anchored inference on frozen H2 checkpoints.

One greedy construction plus three samples. The returned route is the shortest
feasible candidate. Temperature is chosen on the existing synthetic validation
set only. Official Montoya instances are not used here.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import decode, load_policy

OUT = ROOT / "results" / "raw" / "bc_rl_improvement_v1" / "teacher_validation"
RUNS = ROOT / "results" / "raw" / "bc_rl_full_v1" / "runs"
CACHE = ROOT / "results" / "raw" / "bc_rl_full_v1" / "teacher_cache_n10_train500_val100.pkl"
PROTOCOL = {
    "name": "h2_greedy_plus_3_teacher_validation",
    "budget_s": 14400,
    "model": "H2",
    "seeds": [42, 43, 44, 45, 46],
    "roles": ["bc", "rl"],
    "size": 10,
    "n_instances": 100,
    "instance_source": "results/raw/bc_rl_full_v1/teacher_cache_n10_train500_val100.pkl validation split",
    "cache_key": {
        "val_seed_start": 11_200_000,
        "generator": "v2",
        "teacher": "basic",
        "note": "The cache keeps the first 100 seeds at or after val_seed_start whose teacher route is feasible. It is not the raw seed block 11200000-11200099.",
    },
    "temperatures": [1.0, 0.8],
    "n_samples": 3,
    "sample_seed0": 940_000,
    "checkpoints": "results/raw/bc_rl_full_v1/runs/H2_seed{seed}/{bc_best,rl_best}.pt",
    "reselection": False,
    "temperature_scope": "sampled route and customer logits after the feasibility mask; greedy argmax is not scaled",
    "returned_route": "shortest feasible among 1 greedy and 3 samples; ties keep the earlier candidate, greedy first",
    "cost": "sum of the four decode runtimes and frvcpy executions",
    "infeasible_return": "if no candidate is feasible, the penalized duration is 100 h, which is n times T_max",
    "decision_rule": (
        "A temperature is promising if at least 4 of 5 seeds have a negative "
        "RL-minus-BC penalized anchored duration and the paired Student-t 95% interval "
        "lies entirely below zero. Both temperatures are reported. This validation "
        "split was already used for checkpoint selection."
    ),
    "not_used": ["official Montoya instances", "checkpoint reselection", "retraining"],
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "frozen_protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved != PROTOCOL:
            raise SystemExit("frozen protocol already exists and does not match this script")
    else:
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    blob = pickle.loads(CACHE.read_bytes())
    instances = [pair[0] for pair in blob["val"]]
    if len(instances) != PROTOCOL["n_instances"]:
        raise SystemExit(f"expected 100 cached validation instances, found {len(instances)}")
    fields = [
        "model", "seed", "role", "temperature", "instance_index", "instance",
        "protocol", "trial", "feasible", "objective_h", "travel_h", "service_h",
        "charging_h", "runtime_s", "oracle_calls", "oracle_requests",
        "oracle_cache_hits", "oracle_solver_executions",
    ]
    out_path = OUT / "evaluations.csv"
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8", newline="") as handle:
            done = {
                (row["seed"], row["role"], row["temperature"], row["instance_index"], row["protocol"], row["trial"])
                for row in csv.DictReader(handle)
            }
    started = time.perf_counter()
    stopped = False
    from charging import decompose_solution

    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not done:
            writer.writeheader()
        for temperature in PROTOCOL["temperatures"]:
            for seed in PROTOCOL["seeds"]:
                for role, filename in (("bc", "bc_best.pt"), ("rl", "rl_best.pt")):
                    policy = load_policy(RUNS / f"H2_seed{seed}" / filename)
                    for index, instance in enumerate(instances):
                        jobs = [("greedy", 0, False)] + [
                            ("sample", trial, True) for trial in range(PROTOCOL["n_samples"])
                        ]
                        for protocol, trial, sample in jobs:
                            key = (str(seed), role, str(temperature), str(index), protocol, str(trial))
                            if key in done:
                                continue
                            if time.perf_counter() - started > PROTOCOL["budget_s"]:
                                stopped = True
                                break
                            seed_value = 0 if not sample else PROTOCOL["sample_seed0"] + index * 16 + trial
                            with torch.no_grad():
                                result = decode(
                                    policy,
                                    instance,
                                    sample=sample,
                                    seed=seed_value,
                                    temperature=float(temperature),
                                )
                            parts = decompose_solution(instance, result["plans"]) if result["feasible"] else {}
                            writer.writerow(
                                {
                                    "model": "H2",
                                    "seed": seed,
                                    "role": role,
                                    "temperature": temperature,
                                    "instance_index": index,
                                    "instance": instance.name,
                                    "protocol": protocol,
                                    "trial": trial,
                                    "feasible": int(bool(result["feasible"])),
                                    "objective_h": result["objective"] if result["feasible"] else "",
                                    "travel_h": parts.get("travel_h", ""),
                                    "service_h": parts.get("service_h", ""),
                                    "charging_h": parts.get("charging_h", ""),
                                    "runtime_s": result["runtime"],
                                    "oracle_calls": result["oracle_calls"],
                                    "oracle_requests": result["oracle_requests"],
                                    "oracle_cache_hits": result["oracle_cache_hits"],
                                    "oracle_solver_executions": result["oracle_solver_executions"],
                                }
                            )
                            handle.flush()
                        if stopped:
                            break
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break
    status = {
        "completed": not stopped,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "budget_s": PROTOCOL["budget_s"],
        "rows_done_before": len(done),
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(status, flush=True)
    return 2 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
