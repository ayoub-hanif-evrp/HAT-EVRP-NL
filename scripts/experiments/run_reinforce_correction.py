"""One corrected REINFORCE run against BC and original leave-one-out REINFORCE.

The correction is return-to-go from exact nonlinear duration changes. Both RL
runs start from the BC checkpoint trained on this same coverable set. The
protocol file is written before any training update.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import FixedRouteOracle
from data import generate_synthetic
from solver import randomized_greedy

OUT = ROOT / "results" / "raw" / "reinforce_diagnosis_v1"
SEEDS = (42, 43, 44, 45, 46)
DATA_SEED = 481
PROTOCOL = {
    "name": "return_to_go_reinforce_v1",
    "parent_commit": "376cce83c383af00c80cbac3d7da68eb3c2c847f",
    "model": "H2",
    "seeds": list(SEEDS),
    "data_seed": DATA_SEED,
    "size": 10,
    "generator": "v2",
    "n_train_target": 500,
    "train_scan_start": 48_100_000,
    "train_scan_cap": 900,
    "val_seed_start": 49_100_000,
    "n_val": 40,
    "confirm_seed_start": 49_210_000,
    "n_confirm": 40,
    "coverability": "every customer has a feasible dedicated route before the instance can enter training",
    "teacher": "basic randomized greedy, 3 restarts, 1 local-search pass, budget 800; teacher failures are excluded from training and counted",
    "validation_filter": "none; every generated validation instance is kept",
    "bc_steps": 300,
    "rl_steps": 150,
    "eval_every": 50,
    "batch_size": 4,
    "rollouts": 4,
    "bc_lr": 5e-4,
    "rl_lr": 2e-5,
    "grad_clip": 1.0,
    "entropy_coef": 0.0,
    "reward": "normalized",
    "selection": "returned",
    "selection_definition": "0.5 * penalized greedy duration + 0.5 * penalized best feasible among greedy and 3 samples",
    "baselines": {
        "loo": "original leave-one-out terminal REINFORCE",
        "rtg": "exact nonlinear marginal return-to-go with a detached cross-rollout baseline",
    },
    "claim_metric": "greedy penalized duration on all validation instances",
    "practical_threshold_relative": 0.01,
    "decision_rule": (
        "Corrected RL is promising only if its selected greedy duration is at least 1% below "
        "the equally trained BC greedy duration on at least 4 of 5 seeds and the paired "
        "Student-t 95% interval lies entirely below zero. Confirmation is still scored once. "
        "Shift splits are decoded only if that rule passes."
    ),
}


def _teacher(instance, seed: int):
    return randomized_greedy(
        instance,
        restarts=3,
        seed=seed,
        local_search_passes=1,
        local_search_budget=800,
        local_operators="basic",
    )


def _coverable(instance) -> int:
    oracle = FixedRouteOracle(instance)
    return int(all(oracle.solve((customer,)).feasible for customer in instance.customers))


def build_cache() -> None:
    cache = OUT / "teacher_cache.pkl"
    if cache.exists():
        return
    ledger = []
    train = []
    seed = PROTOCOL["train_scan_start"]
    while len(train) < PROTOCOL["n_train_target"] and seed < PROTOCOL["train_scan_start"] + PROTOCOL["train_scan_cap"]:
        instance = generate_synthetic(PROTOCOL["size"], seed, generator_version="v2")
        coverable = _coverable(instance)
        teacher_feasible = 0
        if coverable:
            result = _teacher(instance, seed)
            teacher_feasible = int(bool(result.feasible and result.customer_routes))
            if teacher_feasible:
                train.append((instance, result.customer_routes))
        ledger.append({"split": "train_scan", "seed": seed, "physically_coverable": coverable, "teacher_feasible": teacher_feasible, "used_for_training": int(coverable and teacher_feasible)})
        seed += 1
        if len(train) % 50 == 0 and coverable:
            print({"train_ready": len(train), "scanned": seed - PROTOCOL["train_scan_start"]}, flush=True)
    if len(train) < PROTOCOL["n_train_target"]:
        raise SystemExit(f"only {len(train)} training instances")
    validation = []
    for offset in range(PROTOCOL["n_val"]):
        val_seed = PROTOCOL["val_seed_start"] + offset
        instance = generate_synthetic(PROTOCOL["size"], val_seed, generator_version="v2")
        coverable = _coverable(instance)
        validation.append((instance, tuple()))
        ledger.append({"split": "validation", "seed": val_seed, "physically_coverable": coverable, "teacher_feasible": "", "used_for_training": 0})
    key = {
        "size": PROTOCOL["size"],
        "n_train": PROTOCOL["n_train_target"],
        "n_val": PROTOCOL["n_val"],
        "train_seed_start": 1_100_000 + DATA_SEED * 10_000,
        "val_seed_start": 1_200_000 + DATA_SEED * 10_000,
        "teacher": "basic",
        "generator": "v2",
    }
    cache.write_bytes(pickle.dumps({"key": key, "train": train, "val": validation}))
    with (OUT / "instance_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    print({"cache": str(cache), "train": len(train), "val": len(validation)}, flush=True)


def _command(seed: int, baseline: str, run_id: str, from_bc: str | None) -> list[str]:
    command = [
        sys.executable, "-u", str(ROOT / "train.py"),
        "--pipeline", "primary",
        "--version", "bcrl_rtg_v1",
        "--model", "H2",
        "--size", "10",
        "--seed", str(seed),
        "--data-seed", str(DATA_SEED),
        "--generator", "v2",
        "--train-instances", str(PROTOCOL["n_train_target"]),
        "--val-instances", str(PROTOCOL["n_val"]),
        "--steps", "0" if from_bc else str(PROTOCOL["bc_steps"]),
        "--rl-steps", str(PROTOCOL["rl_steps"]),
        "--eval-every", str(PROTOCOL["eval_every"]),
        "--bc-lr", str(PROTOCOL["bc_lr"]),
        "--rl-lr", str(PROTOCOL["rl_lr"]),
        "--batch-size", str(PROTOCOL["batch_size"]),
        "--rollouts", str(PROTOCOL["rollouts"]),
        "--baseline", baseline,
        "--selection", "returned",
        "--selection-sample-seed0", "981000",
        "--reward", "normalized",
        "--instance-order", "shuffled",
        "--bc-kl", "0",
        "--entropy-coef", "0",
        "--grad-clip", "1.0",
        "--rl-patience", "0",
        "--teacher-cache", str(OUT / "teacher_cache.pkl"),
        "--run-dir", str(OUT / "runs"),
        "--run-id", run_id,
    ]
    if from_bc:
        command.extend(["--rl-from-bc", from_bc])
    return command


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "frozen_protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved != PROTOCOL:
            raise SystemExit("correction protocol already exists and does not match")
    else:
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    build_cache()
    failures = []
    for seed in SEEDS:
        loo_id = f"H2_seed{seed}_loo"
        print({"start": loo_id}, flush=True)
        completed = subprocess.run(_command(seed, "loo", loo_id, None), cwd=ROOT, env=os.environ.copy())
        if completed.returncode != 0:
            failures.append(loo_id)
            print({"failed": loo_id}, flush=True)
            continue
        source = OUT / "runs" / loo_id / "bc_best.pt"
        rtg_id = f"H2_seed{seed}_rtg"
        print({"start": rtg_id}, flush=True)
        completed = subprocess.run(_command(seed, "rtg", rtg_id, str(source)), cwd=ROOT, env=os.environ.copy())
        if completed.returncode != 0:
            failures.append(rtg_id)
            print({"failed": rtg_id}, flush=True)
    print({"failures": failures}, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
