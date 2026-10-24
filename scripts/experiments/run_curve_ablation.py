"""Matched H2 ablation: curve features versus station-type labels only.

Both arms train on the same curve-varied instances, seeds, and BC/RL budget.
Official Montoya curves are not modified. The budget is smaller than the main
study and is not a replacement for those checkpoints.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "raw" / "final_autonomous_study" / "curve_ablation"
CACHE = OUT / "teacher_cache.pkl"
PROTOCOL = {
    "name": "curve_feature_matched_ablation_v1",
    "hypothesis": (
        "Explicit curve-shape features improve greedy returned-route duration on unseen "
        "curve slopes relative to an otherwise identical model whose curve features are zero."
    ),
    "model": "H2",
    "seeds": [42, 43, 44, 45, 46],
    "arms": ["features", "masked"],
    "n_train": 40,
    "n_val": 20,
    "train_seed_start": 37_000_000,
    "val_seed_start": 37_100_000,
    "data_seed": 3700,
    "bc_steps": 40,
    "rl_steps": 40,
    "batch_size": 2,
    "rollouts": 4,
    "baseline": "loo",
    "rl_lr": 2e-5,
    "bc_lr": 5e-4,
    "selection": "greedy penalized duration on the curve-varied validation split",
    "evaluation": "greedy decode on the frozen curve_probe pairs, which are not used for selection",
    "budget_note": "Shorter than the main 1000/300 schedule. The comparison is between the two arms.",
    "teacher_filter": "Behavior cloning keeps a seed only when the teacher route is feasible. Skipped seeds are recorded. Evaluation does not use that filter.",
}


def _builder():
    path = ROOT / "scripts" / "experiments" / "build_unbiased_splits.py"
    spec = importlib.util.spec_from_file_location("build_unbiased_splits", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_cache() -> None:
    if CACHE.exists():
        return
    from data import generate_synthetic
    from solver import randomized_greedy

    builder = _builder()
    skipped = []

    def collect(seed_start: int, count: int):
        rows = []
        seed = seed_start
        while len(rows) < count and seed < seed_start + count * 30:
            instance = builder._perturb_curves(
                generate_synthetic(10, seed, generator_version="v2"),
                seed,
            )
            result = randomized_greedy(
                instance,
                restarts=3,
                seed=seed,
                local_search_passes=1,
                local_search_budget=800,
                local_operators="basic",
            )
            if result.feasible and len(result.customer_routes) >= 1:
                rows.append((instance, result.customer_routes))
            else:
                skipped.append(seed)
            seed += 1
        if len(rows) < count:
            raise SystemExit(f"only built {len(rows)}/{count} teachers from {seed_start}")
        return rows

    train = collect(PROTOCOL["train_seed_start"], PROTOCOL["n_train"])
    validation = collect(PROTOCOL["val_seed_start"], PROTOCOL["n_val"])
    key = {
        "size": 10,
        "n_train": PROTOCOL["n_train"],
        "n_val": PROTOCOL["n_val"],
        "train_seed_start": 1_100_000 + PROTOCOL["data_seed"] * 10_000,
        "val_seed_start": 1_200_000 + PROTOCOL["data_seed"] * 10_000,
        "teacher": "basic",
        "generator": "v2",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps({"key": key, "train": train, "val": validation, "skipped_seeds": skipped}))
    print({"teacher_cache": "wrote", "skipped": len(skipped)}, flush=True)


def command(arm: str, seed: int) -> list[str]:
    cmd = [
        sys.executable, "-u", str(ROOT / "train.py"),
        "--pipeline", "primary",
        "--version", "bcrl_curve_ablation_v1",
        "--model", "H2",
        "--size", "10",
        "--seed", str(seed),
        "--data-seed", str(PROTOCOL["data_seed"]),
        "--generator", "v2",
        "--train-instances", str(PROTOCOL["n_train"]),
        "--val-instances", str(PROTOCOL["n_val"]),
        "--steps", str(PROTOCOL["bc_steps"]),
        "--rl-steps", str(PROTOCOL["rl_steps"]),
        "--eval-every", "20",
        "--bc-lr", "0.0005",
        "--rl-lr", "0.00002",
        "--batch-size", "2",
        "--rollouts", "4",
        "--baseline", "loo",
        "--reward", "normalized",
        "--instance-order", "shuffled",
        "--bc-kl", "0",
        "--entropy-coef", "0",
        "--grad-clip", "1.0",
        "--rl-patience", "0",
        "--bc-patience", "0",
        "--teacher-cache", str(CACHE),
        "--run-dir", str(OUT / "runs" / arm),
        "--run-id", f"H2_seed{seed}",
    ]
    if arm == "masked":
        cmd.append("--mask-curve-features")
    return cmd


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "frozen_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != PROTOCOL:
        raise SystemExit("curve ablation protocol does not match")
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    build_cache()
    failures = []
    for arm in PROTOCOL["arms"]:
        for seed in PROTOCOL["seeds"]:
            print({"start": arm, "seed": seed}, flush=True)
            completed = subprocess.run(command(arm, seed), cwd=ROOT, env=os.environ.copy())
            if completed.returncode != 0:
                failures.append((arm, seed, completed.returncode))
                print({"failed": arm, "seed": seed, "code": completed.returncode}, flush=True)
            else:
                print({"trained": arm, "seed": seed}, flush=True)
    print({"failures": failures}, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
