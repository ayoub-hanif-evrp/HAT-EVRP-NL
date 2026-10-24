"""Compare greedy routes when only charging-curve slopes change.

Station labels and geometry stay fixed. This is an inference diagnostic on the
saved policies. It does not retrain and does not modify official Montoya curves.
"""

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

from train import decode, load_policy

OUT = ROOT / "results" / "raw" / "final_autonomous_study" / "curve_probe_eval"
PROTOCOL = {
    "name": "curve_probe_greedy_inference",
    "budget_s": 3600,
    "split": "curve_probe",
    "candidates": {
        "original_bc": "results/raw/bc_rl_full_v1/runs/H2_seed{seed}/bc_best.pt",
        "original_rl": "results/raw/bc_rl_full_v1/runs/H2_seed{seed}/rl_best.pt",
    },
    "seeds": [42, 43, 44, 45, 46],
    "decode": "greedy",
    "note": "Paired base Montoya curves versus perturbed slopes with the same labels and coordinates.",
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
        raise SystemExit("curve probe protocol does not match")
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    blob = pickle.loads(
        (ROOT / "results" / "raw" / "final_autonomous_study" / "splits" / "instances.pkl").read_bytes()
    )
    builder = _builder()
    instances = blob["splits"]["curve_probe"]
    base = [item for item in instances if not item.name.endswith("_curved")]
    perturbed = {item.name[: -len("_curved")]: item for item in instances if item.name.endswith("_curved")}
    if len(base) != 24 or len(perturbed) != 24:
        raise SystemExit("expected 24 curve pairs")
    fields = [
        "candidate", "seed", "pair_index", "variant", "instance", "feasible",
        "objective_h", "fingerprint", "runtime_s", "oracle_solver_executions",
    ]
    out_path = OUT / "evaluations.csv"
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8", newline="") as handle:
            done = {(row["candidate"], row["seed"], row["pair_index"], row["variant"]) for row in csv.DictReader(handle)}
    started = time.perf_counter()
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not done:
            writer.writeheader()
        for candidate, pattern in PROTOCOL["candidates"].items():
            for seed in PROTOCOL["seeds"]:
                policy = load_policy(ROOT / pattern.format(seed=seed))
                for index, instance in enumerate(base):
                    pair = [("base", instance), ("perturbed_curves", perturbed[instance.name])]
                    for variant, item in pair:
                        key = (candidate, str(seed), str(index), variant)
                        if key in done:
                            continue
                        with torch.no_grad():
                            result = decode(policy, item, sample=False, seed=0)
                        routes = result.get("routes") or ()
                        fingerprint = "|".join("-".join(str(node) for node in route) for route in routes)
                        writer.writerow(
                            {
                                "candidate": candidate,
                                "seed": seed,
                                "pair_index": index,
                                "variant": variant,
                                "instance": item.name,
                                "feasible": int(bool(result["feasible"])),
                                "objective_h": result["objective"] if result["feasible"] else "",
                                "fingerprint": fingerprint,
                                "runtime_s": result["runtime"],
                                "oracle_solver_executions": result["oracle_solver_executions"],
                            }
                        )
                        handle.flush()
    status = {"completed": True, "elapsed_s": round(time.perf_counter() - started, 3)}
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(status, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
