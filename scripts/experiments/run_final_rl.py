"""One final REINFORCE configuration. No further variants.

Route Transformer and RT + DeepSets, seeds 42-46, from the saved BC checkpoints.
Shuffled instance order, 300 updates, batch size 2, four sampled trajectories,
normalized nonlinear reward, gradient clip 1.0, learning rate 2e-5, entropy
coefficient 0, frozen-BC KL coefficient 0. The baseline is a detached greedy
rollout of the live policy. This run does not retrain behavior cloning and does
not write into earlier result directories.

Official evaluation is not decided here. It is a separate locked step on these
checkpoints whether or not validation improves.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "raw" / "bc_rl_full_v1"
OUT = ROOT / "results" / "raw" / "bc_rl_final_v1"
LOGS = OUT / "logs"
CACHE = SOURCE / "teacher_cache_n10_train500_val100.pkl"

MODELS = ("H2", "H4")
SEEDS = (42, 43, 44, 45, 46)
PROTOCOL = {
    "name": "bc_rl_final_v1",
    "hypothesis": (
        "REINFORCE with a detached greedy-rollout baseline and learning rate "
        "2e-5 lowers synthetic validation duration relative to the saved BC "
        "checkpoint, for Route Transformer and RT + DeepSets, seeds 42-46."
    ),
    "models": list(MODELS),
    "seeds": list(SEEDS),
    "rl_steps": 300,
    "rl_lr": 2e-5,
    "baseline": "greedy",
    "bc_kl": 0.0,
    "entropy_coef": 0.0,
    "instance_order": "shuffled",
    "reward": "normalized",
    "rollouts": 4,
    "batch_size": 2,
    "grad_clip": 1.0,
    "does_not_isolate": "greedy baseline and learning rate are changed together",
    "official_eval": "separate locked step, run once even if validation does not improve",
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
        "bcrl_final_v1",
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
        "0.00002",
        "--batch-size",
        "2",
        "--rollouts",
        "4",
        "--baseline",
        "greedy",
        "--reward",
        "normalized",
        "--instance-order",
        "shuffled",
        "--bc-kl",
        "0",
        "--entropy-coef",
        "0",
        "--grad-clip",
        "1.0",
        "--rl-patience",
        "0",
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
            updated_path = OUT / "runs" / f"{model}_seed{seed}" / "run_meta.json"
            if not updated_path.exists():
                continue
            updated = json.loads(updated_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "bc_validation_h": round(_duration(original.get("bc_score")), 4),
                    "final_rl_validation_h": round(_duration(updated.get("rl_best_score")), 4),
                    "final_rl_improved": updated.get("rl_improved_on_validation"),
                    "completed_rl_updates": updated.get("completed_rl_updates"),
                    "baseline": updated.get("baseline"),
                    "rl_lr": updated.get("rl_lr"),
                    "source_bc": str(SOURCE / "runs" / f"{model}_seed{seed}" / "bc_best.pt"),
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


def complete(meta: dict) -> bool:
    return (
        int(meta.get("completed_rl_updates") or 0) >= 300
        and meta.get("baseline") == "greedy"
        and float(meta.get("rl_lr") or 0) == 2e-5
        and float(meta.get("bc_kl") or 0) == 0.0
        and float(meta.get("entropy_coef") or 0) == 0.0
    )


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
            if meta_path.exists() and (run_dir / "rl_best.pt").exists() and (run_dir / "bc_best.pt").exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if complete(meta):
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
    print({"failures": failures, "official_eval": "not started by this script"}, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
