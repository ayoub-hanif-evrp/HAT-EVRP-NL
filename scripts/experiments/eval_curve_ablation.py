"""Greedy evaluation of the matched curve-feature arms on the frozen probe."""

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

OUT = ROOT / "results" / "raw" / "final_autonomous_study" / "curve_ablation" / "probe_eval"
PROTOCOL = {
    "name": "curve_ablation_probe_greedy",
    "arms": ["features", "masked"],
    "seeds": [42, 43, 44, 45, 46],
    "roles": ["bc_best.pt", "rl_best.pt"],
    "probe": "curve_probe",
    "decode": "greedy",
    "selection_not_used": "curve_probe was not used to choose these checkpoints",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "frozen_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != PROTOCOL:
        raise SystemExit("probe protocol does not match")
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    blob = pickle.loads(
        (ROOT / "results" / "raw" / "final_autonomous_study" / "splits" / "instances.pkl").read_bytes()
    )
    instances = blob["splits"]["curve_probe"]
    fields = [
        "arm", "role", "seed", "variant", "pair_index", "instance", "feasible",
        "objective_h", "fingerprint", "runtime_s",
    ]
    out_path = OUT / "evaluations.csv"
    started = time.perf_counter()
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in PROTOCOL["arms"]:
            for role in PROTOCOL["roles"]:
                for seed in PROTOCOL["seeds"]:
                    path = (
                        ROOT
                        / "results"
                        / "raw"
                        / "final_autonomous_study"
                        / "curve_ablation"
                        / "runs"
                        / arm
                        / f"H2_seed{seed}"
                        / role
                    )
                    policy = load_policy(path)
                    for index, instance in enumerate(instances):
                        variant = "perturbed_curves" if instance.name.endswith("_curved") else "base"
                        pair_index = index // 2
                        with torch.no_grad():
                            result = decode(policy, instance, sample=False, seed=0)
                        routes = result.get("routes") or ()
                        writer.writerow(
                            {
                                "arm": arm,
                                "role": role,
                                "seed": seed,
                                "variant": variant,
                                "pair_index": pair_index,
                                "instance": instance.name,
                                "feasible": int(bool(result["feasible"])),
                                "objective_h": result["objective"] if result["feasible"] else "",
                                "fingerprint": "|".join("-".join(str(node) for node in route) for route in routes),
                                "runtime_s": result["runtime"],
                            }
                        )
    print({"completed": True, "elapsed_s": round(time.perf_counter() - started, 3), "rows": len(instances) * 2 * 2 * 5}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
