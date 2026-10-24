"""One frozen slow-charger holdout. Evaluation only.

The protocol file is written before the first decode. The run stops if it
exceeds the saved wall-clock budget. Checkpoints are not reselected.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import SyntheticConfig, generate_synthetic
from train import decode, load_policy

OUT = ROOT / "results" / "raw" / "shift_holdout_v1"
RUNS = ROOT / "results" / "raw" / "bc_rl_full_v1" / "runs"
PROTOCOL = {
    "name": "slow_only_clustered_pmedian",
    "budget_s": 480,
    "model": "H2",
    "seeds": [42, 43, 44, 45, 46],
    "size": 10,
    "n_instances": 20,
    "instance_seed0": 21_300_000,
    "sample_seed0": 930_000,
    "generator_version": "v2",
    "config": {
        "geometry": "clustered",
        "station_placement": "pmedian_region",
        "tech_pool": ["slow"],
        "station_count": 2,
        "depot_bias": "center",
        "density": 0.2,
        "cluster_sigma": 6.0,
        "layout_family": "compact",
        "region_span": 120.0,
        "n_clusters": 3,
    },
    "k_samples": 4,
    "checkpoints": "results/raw/bc_rl_full_v1/runs/H2_seed{seed}/{bc_best,rl_best}.pt",
    "reselection": False,
    "disjoint_from": {
        "training_data_seed": 1000,
        "train_instance_seed_start": 11_100_000,
        "validation_seed_start": 11_200_000,
        "official_sample_seed0": 910_000,
    },
}


def _config() -> SyntheticConfig:
    raw = PROTOCOL["config"]
    return SyntheticConfig(
        geometry=raw["geometry"],
        station_placement=raw["station_placement"],
        density=raw["density"],
        tech_pool=tuple(raw["tech_pool"]),
        cluster_sigma=raw["cluster_sigma"],
        n_clusters=raw["n_clusters"],
        station_count=raw["station_count"],
        depot_bias=raw["depot_bias"],
        layout_family=raw["layout_family"],
        region_span=raw["region_span"],
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "frozen_protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved != PROTOCOL:
            raise SystemExit("frozen protocol already exists and does not match this script")
    else:
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    config = _config()
    instances = [
        generate_synthetic(
            PROTOCOL["size"],
            PROTOCOL["instance_seed0"] + index,
            config=config,
            generator_version=PROTOCOL["generator_version"],
        )
        for index in range(PROTOCOL["n_instances"])
    ]
    techs = sorted({instance.nodes[station].station_type for instance in instances for station in instance.stations})
    if techs != ["slow"]:
        raise SystemExit(f"shift did not produce slow-only stations: {techs}")
    fields = [
        "model", "seed", "role", "instance_index", "protocol", "trial",
        "feasible", "objective_h", "runtime_s", "oracle_calls", "oracle_requests", "oracle_cache_hits",
    ]
    out_path = OUT / "evaluations.csv"
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8", newline="") as handle:
            done = {
                (row["seed"], row["role"], row["instance_index"], row["protocol"], row["trial"])
                for row in csv.DictReader(handle)
            }
    started = time.perf_counter()
    policies = {}
    stopped = False
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not done:
            writer.writeheader()
        for seed in PROTOCOL["seeds"]:
            for role, filename in (("bc", "bc_best.pt"), ("rl", "rl_best.pt")):
                policies[role] = load_policy(RUNS / f"H2_seed{seed}" / filename)
                for index, instance in enumerate(instances):
                    jobs = [("greedy", 0, False, 0)]
                    for trial in range(PROTOCOL["k_samples"]):
                        jobs.append(("sample", trial, True, PROTOCOL["sample_seed0"] + index * 16 + trial))
                    for protocol, trial, sample, decode_seed in jobs:
                        key = (str(seed), role, str(index), protocol, str(trial))
                        if key in done:
                            continue
                        if time.perf_counter() - started > PROTOCOL["budget_s"]:
                            stopped = True
                            break
                        with torch.no_grad():
                            result = decode(policies[role], instance, sample=sample, seed=decode_seed)
                        writer.writerow(
                            {
                                "model": "H2",
                                "seed": seed,
                                "role": role,
                                "instance_index": index,
                                "protocol": protocol,
                                "trial": trial,
                                "feasible": int(bool(result["feasible"])),
                                "objective_h": result["objective"] if result["feasible"] else "",
                                "runtime_s": result["runtime"],
                                "oracle_calls": result["oracle_calls"],
                                "oracle_requests": result["oracle_requests"],
                                "oracle_cache_hits": result["oracle_cache_hits"],
                            }
                        )
                        handle.flush()
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
        "station_technologies": techs,
        "n_instances": len(instances),
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(status)
    return 0 if not stopped else 2


if __name__ == "__main__":
    raise SystemExit(main())
