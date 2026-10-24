"""Score the objective-aligned checkpoints on the frozen unbiased splits."""

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

STUDY = ROOT / "results" / "raw" / "final_autonomous_study"
OUT = STUDY / "objective_aligned" / "anchored_eval"
PROTOCOL = {
    "name": "objective_aligned_unbiased_eval",
    "budget_s": 14400,
    "splits": ["dev_v2", "confirm_v2", "shift_v2c", "shift_slow", "shift_geo"],
    "candidates": {
        "original_bc": "results/raw/bc_rl_full_v1/runs/H2_seed{seed}/bc_best.pt",
        "objective_aligned_rl": "results/raw/final_autonomous_study/objective_aligned/runs/H2_seed{seed}/rl_best.pt",
    },
    "seeds": [42, 43, 44, 45, 46],
    "n_samples": 3,
    "temperature": 1.0,
    "selection_note": "dev_v2 sample seeds match checkpoint selection. confirm_v2 and shift splits were not used to choose the checkpoint.",
}


def _builder():
    path = ROOT / "scripts" / "experiments" / "build_unbiased_splits.py"
    spec = importlib.util.spec_from_file_location("build_unbiased_splits", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "frozen_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != PROTOCOL:
        raise SystemExit("objective eval protocol does not match")
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    blob = pickle.loads((STUDY / "splits" / "instances.pkl").read_bytes())
    builder = _builder()
    hashes = {
        (row["split"], row["name"]): row["sha256"]
        for row in csv.DictReader((STUDY / "splits" / "hashes.csv").open(encoding="utf-8"))
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
            spec_row = next(item for item in blob["protocol"]["splits"] if item["name"] == split_name)
            instances = [item for item in blob["splits"][split_name] if not str(item.name).endswith("_curved")]
            for instance in instances:
                if hashes[(split_name, instance.name)] != builder._instance_hash(instance):
                    raise SystemExit(f"hash mismatch for {instance.name}")
            for candidate, pattern in PROTOCOL["candidates"].items():
                for seed in PROTOCOL["seeds"]:
                    policy = load_policy(ROOT / pattern.format(seed=seed))
                    for index, instance in enumerate(instances):
                        jobs = [("greedy", 0, False)] + [("sample", trial, True) for trial in range(3)]
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
                                    seed=0 if not sample else spec_row["sample_seed0"] + index * 16 + trial,
                                    temperature=1.0,
                                )
                            parts = decompose_solution(instance, result["plans"]) if result["feasible"] else {}
                            routes = result.get("routes") or ()
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
                                    "customer_fingerprint": "|".join(
                                        "-".join(str(node) for node in route) for route in routes
                                    ),
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
