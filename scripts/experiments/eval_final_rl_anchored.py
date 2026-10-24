"""Score the greedy-baseline checkpoints with one greedy decode plus three samples.

Temperature is 1, the training temperature. This does not retune on Montoya instances.
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

from charging import decompose_solution
from train import decode, load_policy

OUT = ROOT / "results" / "raw" / "bc_rl_final_v1" / "anchored_validation"
RUNS = ROOT / "results" / "raw" / "bc_rl_final_v1" / "runs"
CACHE = ROOT / "results" / "raw" / "bc_rl_full_v1" / "teacher_cache_n10_train500_val100.pkl"
PROTOCOL = {
    "name": "greedy_baseline_anchored_teacher_validation",
    "budget_s": 7200,
    "models": ["H2", "H4"],
    "seeds": [42, 43, 44, 45, 46],
    "roles": ["bc", "rl"],
    "size": 10,
    "n_instances": 100,
    "instance_source": "results/raw/bc_rl_full_v1/teacher_cache_n10_train500_val100.pkl validation split",
    "temperature": 1.0,
    "n_samples": 3,
    "sample_seed0": 940_000,
    "checkpoints": "results/raw/bc_rl_final_v1/runs/{model}_seed{seed}/{bc_best,rl_best}.pt",
    "reselection": False,
    "note": "Same teacher-feasible validation instances as training. RL weights are the greedy-rollout checkpoints.",
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
    if len(instances) != 100:
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
                (row["model"], row["seed"], row["role"], row["instance_index"], row["protocol"], row["trial"])
                for row in csv.DictReader(handle)
            }
    started = time.perf_counter()
    stopped = False
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not done:
            writer.writeheader()
        for model in PROTOCOL["models"]:
            for seed in PROTOCOL["seeds"]:
                for role, filename in (("bc", "bc_best.pt"), ("rl", "rl_best.pt")):
                    policy = load_policy(RUNS / f"{model}_seed{seed}" / filename)
                    for index, instance in enumerate(instances):
                        jobs = [("greedy", 0, False)] + [("sample", trial, True) for trial in range(3)]
                        for protocol, trial, sample in jobs:
                            key = (model, str(seed), role, str(index), protocol, str(trial))
                            if key in done:
                                continue
                            if time.perf_counter() - started > PROTOCOL["budget_s"]:
                                stopped = True
                                break
                            with torch.no_grad():
                                result = decode(
                                    policy,
                                    instance,
                                    sample=sample,
                                    seed=0 if not sample else PROTOCOL["sample_seed0"] + index * 16 + trial,
                                    temperature=1.0,
                                )
                            parts = decompose_solution(instance, result["plans"]) if result["feasible"] else {}
                            writer.writerow(
                                {
                                    "model": model,
                                    "seed": seed,
                                    "role": role,
                                    "temperature": 1.0,
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
    status = {"completed": not stopped, "elapsed_s": round(time.perf_counter() - started, 3)}
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(status, flush=True)
    return 2 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
