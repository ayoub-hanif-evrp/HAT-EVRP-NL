"""Construction-only audit of saved BC and RL checkpoints.

Frozen before any row is written. A credible RL benefit on one model, one
existing experiment, and one inference protocol requires both of the following
on the shared synthetic validation set:

- the RL penalized duration is lower on at least 4 of 5 seeds
- the Student-t 95% interval of the paired seed differences (RL - BC) lies
  entirely below zero

Protocols are greedy construction, the mean of K=4 sampled constructions, and
the best feasible construction among those same four. Best-of-four is not
reported as greedy. No local search, ruin-and-repair, or hybrid insertion.

If no cell meets the rule, the separate final REINFORCE run is the only
further training. This script does not train.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import decode, load_policy

OUT = ROOT / "results" / "raw" / "inference_audit_v1"
CACHE = ROOT / "results" / "raw" / "bc_rl_full_v1" / "teacher_cache_n10_train500_val100.pkl"
SOURCE = {
    "original": ROOT / "results" / "raw" / "bc_rl_full_v1" / "runs",
    "shuffled": ROOT / "results" / "raw" / "bc_rl_shuffle_v1" / "runs",
    "kl_v2": ROOT / "results" / "raw" / "bc_rl_kl_v2" / "runs",
}
EXPERIMENT_MODELS = {
    "original": ("H1", "H2", "H3", "H4"),
    "shuffled": ("H2", "H4"),
    "kl_v2": ("H2", "H4"),
}
SEEDS = (42, 43, 44, 45, 46)
K = 4
SAMPLE_SEED0 = 910_000
T_CRIT_DF4 = 2.7764451051977987
PROTOCOL = {
    "validation": "synthetic n=10, data_seed 1000, val_seed_start 11200000, 100 instances",
    "k": K,
    "sample_seed": "910000 + instance_index * 16 + trial",
    "same_seeds_for_bc_and_rl": True,
    "objective": "Z_full = travel + service + nonlinear charging",
    "infeasible_hours": "n * T_max",
    "best_of_k": "minimum feasible sample, else the infeasible cap",
    "sampled_mean": "mean of the K penalized durations",
    "search": "none",
    "credible_benefit": (
        "At least 4 of 5 seeds have RL duration lower than BC, and the paired "
        "95% interval of RL-BC is entirely below zero."
    ),
}

FIELDS = [
    "experiment",
    "model",
    "seed",
    "role",
    "checkpoint",
    "instance_index",
    "instance_name",
    "greedy_feasible",
    "greedy_h",
    "greedy_runtime_s",
    "greedy_oracle_calls",
    "sample0_feasible",
    "sample0_h",
    "sample1_feasible",
    "sample1_h",
    "sample2_feasible",
    "sample2_h",
    "sample3_feasible",
    "sample3_h",
    "sample_oracle_calls",
    "sample_runtime_s",
    "sampled_mean_h",
    "sampled_feasible_rate",
    "best_of_4_h",
    "best_of_4_feasible",
]


def _hours(result: dict, cap: float) -> tuple[float, int]:
    if result.get("feasible") and math.isfinite(float(result["objective"])):
        return float(result["objective"]), 1
    return cap, 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validation(limit: int | None) -> list:
    blob = pickle.loads(CACHE.read_bytes())
    key = blob.get("key") or {}
    if int(key.get("val_seed_start", -1)) != 11_200_000 or int(key.get("n_val", -1)) != 100:
        raise SystemExit(f"unexpected validation cache key: {key}")
    instances = [pair[0] for pair in blob["val"]]
    if limit is not None:
        instances = instances[:limit]
    return instances


def jobs(experiments: list[str], models: list[str] | None, seeds: list[int] | None) -> list[dict]:
    chosen_seeds = tuple(seeds) if seeds else SEEDS
    rows = []
    # Behavior-cloning checkpoints live in the original matrix. Later runs load those weights.
    for model in EXPERIMENT_MODELS["original"]:
        if models and model not in models:
            continue
        for seed in chosen_seeds:
            path = SOURCE["original"] / f"{model}_seed{seed}" / "bc_best.pt"
            rows.append({"experiment": "original", "model": model, "seed": seed, "role": "bc", "path": path})
    for experiment in experiments:
        for model in EXPERIMENT_MODELS[experiment]:
            if models and model not in models:
                continue
            for seed in chosen_seeds:
                path = SOURCE[experiment] / f"{model}_seed{seed}" / "rl_best.pt"
                rows.append(
                    {"experiment": experiment, "model": model, "seed": seed, "role": "rl", "path": path}
                )
    return rows


def _done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["experiment"], row["model"], row["seed"], row["role"], row["instance_index"])
            for row in csv.DictReader(handle)
        }


def evaluate_job(job: dict, instances: list, writer, done: set[tuple], handle) -> None:
    if not job["path"].exists():
        raise SystemExit(f"missing checkpoint {job['path']}")
    model = load_policy(job["path"])
    for index, instance in enumerate(instances):
        key = (job["experiment"], job["model"], str(job["seed"]), job["role"], str(index))
        if key in done:
            continue
        cap = len(instance.customers) * float(instance.max_route_hours)
        with torch.no_grad():
            greedy = decode(model, instance, sample=False, seed=0)
            greedy_h, greedy_ok = _hours(greedy, cap)
            samples = []
            sample_calls = 0
            sample_runtime = 0.0
            for trial in range(K):
                sampled = decode(
                    model,
                    instance,
                    sample=True,
                    seed=SAMPLE_SEED0 + index * 16 + trial,
                )
                samples.append(_hours(sampled, cap))
                sample_calls += int(sampled.get("oracle_calls") or 0)
                sample_runtime += float(sampled.get("runtime") or 0.0)
        feasible_hours = [hours for hours, ok in samples if ok]
        sampled_mean = sum(hours for hours, _ok in samples) / K
        if feasible_hours:
            best = min(feasible_hours)
            best_ok = 1
        else:
            best = cap
            best_ok = 0
        row = {
            "experiment": job["experiment"],
            "model": job["model"],
            "seed": job["seed"],
            "role": job["role"],
            "checkpoint": str(job["path"].relative_to(ROOT)),
            "instance_index": index,
            "instance_name": getattr(instance, "name", ""),
            "greedy_feasible": greedy_ok,
            "greedy_h": greedy_h,
            "greedy_runtime_s": greedy.get("runtime", ""),
            "greedy_oracle_calls": greedy.get("oracle_calls", ""),
            "sample_oracle_calls": sample_calls,
            "sample_runtime_s": sample_runtime,
            "sampled_mean_h": sampled_mean,
            "sampled_feasible_rate": sum(ok for _hours, ok in samples) / K,
            "best_of_4_h": best,
            "best_of_4_feasible": best_ok,
        }
        for trial, (hours, ok) in enumerate(samples):
            row[f"sample{trial}_feasible"] = ok
            row[f"sample{trial}_h"] = hours
        writer.writerow(row)
        handle.flush()
        done.add(key)
    print({"evaluated": f"{job['experiment']}:{job['model']}_seed{job['seed']}:{job['role']}"}, flush=True)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(instance_csv: Path) -> list[dict]:
    with instance_csv.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    grouped: dict[tuple, list[dict]] = {}
    for row in raw:
        grouped.setdefault((row["experiment"], row["model"], row["seed"], row["role"]), []).append(row)
    checkpoint_rows = []
    for (experiment, model, seed, role), rows in sorted(grouped.items()):
        checkpoint_rows.append(
            {
                "experiment": experiment,
                "model": model,
                "seed": int(seed),
                "role": role,
                "n_instances": len(rows),
                "greedy_h": _mean([float(row["greedy_h"]) for row in rows]),
                "sampled_mean_h": _mean([float(row["sampled_mean_h"]) for row in rows]),
                "best_of_4_h": _mean([float(row["best_of_4_h"]) for row in rows]),
                "greedy_feasible": _mean([float(row["greedy_feasible"]) for row in rows]),
                "sampled_feasible": _mean([float(row["sampled_feasible_rate"]) for row in rows]),
                "best_of_4_feasible": _mean([float(row["best_of_4_feasible"]) for row in rows]),
                "greedy_oracle_calls": sum(float(row["greedy_oracle_calls"]) for row in rows),
                "sample_oracle_calls": sum(float(row["sample_oracle_calls"]) for row in rows),
                "greedy_runtime_s": sum(float(row["greedy_runtime_s"]) for row in rows),
                "sample_runtime_s": sum(float(row["sample_runtime_s"]) for row in rows),
            }
        )
    return checkpoint_rows


def paired_decision(checkpoint_rows: list[dict], n_instances: int) -> list[dict]:
    """Pair each RL checkpoint with original-matrix BC on the same model and seed."""
    bc = {
        (row["model"], row["seed"]): row
        for row in checkpoint_rows
        if row["experiment"] == "original" and row["role"] == "bc" and row["n_instances"] == n_instances
    }
    cells = []
    protocols = ("greedy_h", "sampled_mean_h", "best_of_4_h")
    by_cell: dict[tuple, list] = {}
    for row in checkpoint_rows:
        if row["role"] != "rl" or row["n_instances"] != n_instances:
            continue
        base = bc.get((row["model"], row["seed"]))
        if base is None:
            continue
        for protocol in protocols:
            by_cell.setdefault((row["experiment"], row["model"], protocol), []).append(
                row[protocol] - base[protocol]
            )
    for (experiment, model, protocol), deltas in sorted(by_cell.items()):
        if len(deltas) < 5:
            cells.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "protocol": protocol,
                    "n_seeds": len(deltas),
                    "credible_benefit": False,
                    "reason": "fewer than 5 seeds",
                }
            )
            continue
        mean = _mean(deltas)
        var = sum((delta - mean) ** 2 for delta in deltas) / (len(deltas) - 1)
        half = T_CRIT_DF4 * math.sqrt(var / len(deltas))
        better = sum(delta < -1e-4 for delta in deltas)
        benefit = better >= 4 and mean + half < 0
        cells.append(
            {
                "experiment": experiment,
                "model": model,
                "protocol": protocol,
                "n_seeds": len(deltas),
                "mean_delta_h": round(mean, 4),
                "ci_low_h": round(mean - half, 4),
                "ci_high_h": round(mean + half, 4),
                "seeds_better": better,
                "credible_benefit": benefit,
            }
        )
    return cells


def write_summary(checkpoint_rows: list[dict], cells: list[dict]) -> None:
    summary_path = OUT / "checkpoint_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0]))
        writer.writeheader()
        for row in checkpoint_rows:
            stored = dict(row)
            for key, value in stored.items():
                if isinstance(value, float):
                    stored[key] = round(value, 6)
            writer.writerow(stored)
    decision = {
        "rule": PROTOCOL["credible_benefit"],
        "cells": cells,
        "any_credible_benefit": any(cell.get("credible_benefit") for cell in cells),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(decision, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-instances", type=int, default=0)
    parser.add_argument("--experiments", nargs="*", default=list(SOURCE))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "frozen_protocol.json"
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    instance_csv = OUT / "instances.csv"
    if args.summarize_only:
        rows = summarize(instance_csv)
        limit = args.limit_instances or 100
        write_summary(rows, paired_decision(rows, limit if args.limit_instances else 100))
        return 0
    instances = load_validation(args.limit_instances or None)
    planned = jobs(args.experiments, args.models, args.seeds)
    manifest = []
    for job in planned:
        manifest.append(
            {
                **{key: job[key] for key in ("experiment", "model", "seed", "role")},
                "checkpoint": str(job["path"].relative_to(ROOT)),
                "sha256": _sha256(job["path"]) if job["path"].exists() else "MISSING",
            }
        )
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    done = _done(instance_csv)
    new_file = not instance_csv.exists()
    with instance_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for job in planned:
            evaluate_job(job, instances, writer, done, handle)
    rows = summarize(instance_csv)
    complete = [row for row in rows if row["n_instances"] == len(instances)]
    if complete:
        write_summary(complete, paired_decision(complete, len(instances)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
