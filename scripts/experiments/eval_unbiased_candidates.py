"""Score frozen checkpoints on the unbiased splits. Resumes from evaluations.csv."""

from __future__ import annotations

import csv
import importlib.util
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

builder_path = ROOT / "scripts" / "experiments" / "build_unbiased_splits.py"
spec = importlib.util.spec_from_file_location("build_unbiased_splits", builder_path)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

STUDY = ROOT / "results" / "raw" / "final_autonomous_study"
SPILTS = STUDY / "splits"
OUT = STUDY / "unbiased_eval"
PROTOCOL = {
    "name": "existing_checkpoints_unbiased_eval",
    "budget_s": 14400,
    "instance_source": "results/raw/final_autonomous_study/splits/instances.pkl",
    "splits": ["dev_v2", "confirm_v2", "shift_v2c", "shift_slow", "shift_geo"],
    "candidates": {
        "original_bc": "results/raw/bc_rl_full_v1/runs/H2_seed{seed}/bc_best.pt",
        "original_rl": "results/raw/bc_rl_full_v1/runs/H2_seed{seed}/rl_best.pt",
        "greedy_baseline_rl": "results/raw/bc_rl_final_v1/runs/H2_seed{seed}/rl_best.pt",
    },
    "seeds": [42, 43, 44, 45, 46],
    "n_samples": 3,
    "temperature": 1.0,
    "returned_route": "shortest feasible among 1 greedy and 3 samples; ties keep greedy first",
    "penalized_cap": "n_customers * max_route_hours",
    "primary_metric": "penalized returned-route duration",
    "decision_use": "Describe existing checkpoints. Do not select a new checkpoint from confirm_v2 or the shift splits.",
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
    blob = pickle.loads((SPILTS / "instances.pkl").read_bytes())
    hashes = {
        (row["split"], row["name"]): row["sha256"]
        for row in csv.DictReader((SPILTS / "hashes.csv").open(encoding="utf-8"))
    }

    fields = [
        "split", "candidate", "seed", "instance_index", "instance", "sha256",
        "protocol", "trial", "feasible", "objective_h", "travel_h", "service_h",
        "charging_h", "runtime_s", "oracle_solver_executions", "customer_fingerprint",
    ]
    out_path = OUT / "evaluations.csv"
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8", newline="") as handle:
            done = {
                (row["split"], row["candidate"], row["seed"], row["instance_index"], row["protocol"], row["trial"])
                for row in csv.DictReader(handle)
            }
    started = time.perf_counter()
    stopped = False
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not done:
            writer.writeheader()
        for split_name in PROTOCOL["splits"]:
            spec = next(item for item in blob["protocol"]["splits"] if item["name"] == split_name)
            instances = [item for item in blob["splits"][split_name] if not str(item.name).endswith("_curved")]
            if len(instances) != spec["n"]:
                raise SystemExit(f"{split_name} has {len(instances)} instances, expected {spec['n']}")
            for instance in instances:
                digest = builder._instance_hash(instance)
                if hashes[(split_name, instance.name)] != digest:
                    raise SystemExit(f"hash mismatch for {instance.name}")
            for candidate, pattern in PROTOCOL["candidates"].items():
                for seed in PROTOCOL["seeds"]:
                    policy = load_policy(ROOT / pattern.format(seed=seed))
                    for index, instance in enumerate(instances):
                        jobs = [("greedy", 0, False)] + [
                            ("sample", trial, True) for trial in range(PROTOCOL["n_samples"])
                        ]
                        for protocol, trial, sample in jobs:
                            key = (split_name, candidate, str(seed), str(index), protocol, str(trial))
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
                                    seed=0 if not sample else spec["sample_seed0"] + index * 16 + trial,
                                    temperature=PROTOCOL["temperature"],
                                )
                            parts = decompose_solution(instance, result["plans"]) if result["feasible"] else {}
                            sequences = result.get("routes") or ()
                            fingerprint = "|".join("-".join(str(node) for node in route) for route in sequences)
                            writer.writerow(
                                {
                                    "split": split_name,
                                    "candidate": candidate,
                                    "seed": seed,
                                    "instance_index": index,
                                    "instance": instance.name,
                                    "sha256": hashes[(split_name, instance.name)],
                                    "protocol": protocol,
                                    "trial": trial,
                                    "feasible": int(bool(result["feasible"])),
                                    "objective_h": result["objective"] if result["feasible"] else "",
                                    "travel_h": parts.get("travel_h", ""),
                                    "service_h": parts.get("service_h", ""),
                                    "charging_h": parts.get("charging_h", ""),
                                    "runtime_s": result["runtime"],
                                    "oracle_solver_executions": result["oracle_solver_executions"],
                                    "customer_fingerprint": fingerprint,
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
