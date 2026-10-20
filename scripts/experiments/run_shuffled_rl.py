"""REINFORCE from the saved BC checkpoints with a full-dataset shuffle.

Does not retrain behavior cloning and does not write into bc_rl_full_v1.
Primary comparison, frozen before these runs: validation penalized duration
of shuffled RL versus the original cyclic RL, for Route Transformer and
RT + DeepSets, seeds 42-46. Official P0/P1/P2 evaluation uses the same
protocol as the original matrix and is not used to choose the sampler.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "raw" / "bc_rl_full_v1"
OUT = ROOT / "results" / "raw" / "bc_rl_shuffle_v1"
LOGS = OUT / "logs"
CACHE = SOURCE / "teacher_cache_n10_train500_val100.pkl"

MODELS = ("H2", "H4")
SEEDS = (42, 43, 44, 45, 46)
PROTOCOL = {
    "hypothesis": (
        "Shuffled RL, started from the original bc_best weights, has a lower "
        "synthetic validation duration than the original cyclic RL."
    ),
    "models": list(MODELS),
    "seeds": list(SEEDS),
    "rl_steps": 300,
    "batch_size": 2,
    "rollouts": 4,
    "baseline": "loo",
    "reward": "normalized",
    "rl_lr": 5e-5,
    "instance_order": "shuffled",
    "data_seed": 1000,
    "official_eval": "same P0/P1/P2 protocol; not used for selection",
}


def command(model: str, seed: int) -> list[str]:
    run_id = f"{model}_seed{seed}"
    source = SOURCE / "runs" / run_id / "bc_best.pt"
    return [
        sys.executable,
        "-u",
        str(ROOT / "train.py"),
        "--pipeline",
        "primary",
        "--version",
        "bcrl_shuffle_v1",
        "--model",
        model,
        "--size",
        "10",
        "--seed",
        str(seed),
        "--data-seed",
        "1000",
        "--generator",
        "v2",
        "--train-instances",
        "500",
        "--val-instances",
        "100",
        "--steps",
        "0",
        "--rl-steps",
        "300",
        "--eval-every",
        "25",
        "--bc-lr",
        "0.0005",
        "--rl-lr",
        "0.00005",
        "--batch-size",
        "2",
        "--rollouts",
        "4",
        "--baseline",
        "loo",
        "--reward",
        "normalized",
        "--instance-order",
        "shuffled",
        "--rl-from-bc",
        str(source),
        "--teacher-cache",
        str(CACHE),
        "--run-dir",
        str(OUT / "runs"),
        "--run-id",
        run_id,
    ]


def _duration(score) -> float:
    if not score or len(score) < 2:
        return float("nan")
    return -float(score[1])


def write_validation_comparison() -> None:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            original = json.loads((SOURCE / "runs" / f"{model}_seed{seed}" / "run_meta.json").read_text(encoding="utf-8"))
            new_path = OUT / "runs" / f"{model}_seed{seed}" / "run_meta.json"
            if not new_path.exists():
                continue
            updated = json.loads(new_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "bc_validation_h": round(_duration(original.get("bc_score")), 4),
                    "original_rl_validation_h": round(_duration(original.get("rl_best_score")), 4),
                    "shuffled_rl_validation_h": round(_duration(updated.get("rl_best_score")), 4),
                    "original_rl_improved": original.get("rl_improved_on_validation"),
                    "shuffled_rl_improved": updated.get("rl_improved_on_validation"),
                    "shuffled_rl_updates": updated.get("completed_rl_updates"),
                }
            )
    path = OUT / "validation_comparison.csv"
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print({"validation_comparison": str(path)}, flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_protocol.json").write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    failures = 0
    for model in MODELS:
        for seed in SEEDS:
            run_id = f"{model}_seed{seed}"
            run_dir = OUT / "runs" / run_id
            meta_path = run_dir / "run_meta.json"
            if meta_path.exists() and (run_dir / "rl_best.pt").exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if int(meta.get("completed_rl_updates") or 0) >= 300 and meta.get("instance_order") == "shuffled":
                    print({"skip_complete": run_id}, flush=True)
                    continue
            source = SOURCE / "runs" / run_id / "bc_best.pt"
            if not source.exists():
                print({"missing_bc": str(source)}, flush=True)
                failures += 1
                continue
            cmd = command(model, seed)
            (LOGS / f"{run_id}.cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
            print({"start": run_id}, flush=True)
            with (LOGS / f"{run_id}.log").open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            if proc.returncode != 0:
                failures += 1
                print({"failed": run_id, "exit": proc.returncode}, flush=True)
            else:
                print({"trained": run_id}, flush=True)
    write_validation_comparison()
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
    from eval_bc_rl_official import evaluate_run

    for model in MODELS:
        for seed in SEEDS:
            run_dir = OUT / "runs" / f"{model}_seed{seed}"
            out_csv = OUT / "eval" / f"{model}_seed{seed}.csv"
            if not (run_dir / "rl_best.pt").exists():
                failures += 1
                continue
            if out_csv.exists() and out_csv.stat().st_size > 0:
                print({"eval_exists": out_csv.name}, flush=True)
                continue
            print({"eval": f"{model}_seed{seed}"}, flush=True)
            evaluate_run(run_dir, ROOT / "dataset", out_csv)
    print({"failures": failures}, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
