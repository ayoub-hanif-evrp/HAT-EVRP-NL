"""REINFORCE from saved BC checkpoints with a distribution KL penalty.

The penalty is KL(pi_theta || pi_BC) on the masked route distribution and the
masked customer distribution. It replaces the earlier sampled log-ratio, whose
expected gradient was zero. This run does not retrain BC, does not write into
bc_rl_kl_v1, and does not run ruin-and-repair. Official neural-construction
evaluation starts only if validation supports it.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "raw" / "bc_rl_full_v1"
SHUFFLE = ROOT / "results" / "raw" / "bc_rl_shuffle_v1"
OUT = ROOT / "results" / "raw" / "bc_rl_kl_v2"
LOGS = OUT / "logs"
CACHE = SOURCE / "teacher_cache_n10_train500_val100.pkl"
KL = "0.01"

MODELS = ("H2", "H4")
SEEDS = (42, 43, 44, 45, 46)
PROTOCOL = {
    "hypothesis": (
        "A 0.01 KL between the live and frozen-BC action distributions lowers "
        "synthetic validation duration relative to the BC checkpoint, for "
        "Route Transformer and RT + DeepSets, seeds 42-46."
    ),
    "models": list(MODELS),
    "seeds": list(SEEDS),
    "rl_steps": 300,
    "bc_kl": 0.01,
    "bc_kl_gradient": "distribution",
    "entropy_coef": 0.0,
    "instance_order": "shuffled",
    "reward": "normalized",
    "baseline": "loo",
    "rl_lr": 5e-5,
    "official_eval": "neural-construction only, and only if validation improves",
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
        "bcrl_kl_v2",
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
        "--bc-kl",
        KL,
        "--entropy-coef",
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


def write_validation_comparison() -> list[dict]:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            original = json.loads((SOURCE / "runs" / f"{model}_seed{seed}" / "run_meta.json").read_text(encoding="utf-8"))
            shuffled_path = SHUFFLE / "runs" / f"{model}_seed{seed}" / "run_meta.json"
            shuffled = json.loads(shuffled_path.read_text(encoding="utf-8")) if shuffled_path.exists() else {}
            updated_path = OUT / "runs" / f"{model}_seed{seed}" / "run_meta.json"
            if not updated_path.exists():
                continue
            updated = json.loads(updated_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "bc_validation_h": round(_duration(original.get("bc_score")), 4),
                    "original_rl_validation_h": round(_duration(original.get("rl_best_score")), 4),
                    "shuffled_rl_validation_h": round(_duration(shuffled.get("rl_best_score")), 4) if shuffled else "",
                    "kl_rl_validation_h": round(_duration(updated.get("rl_best_score")), 4),
                    "kl_rl_improved": updated.get("rl_improved_on_validation"),
                    "bc_kl": updated.get("bc_kl"),
                }
            )
    path = OUT / "validation_comparison.csv"
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print({"validation_comparison": str(path)}, flush=True)
    return rows


def validation_supports_official_eval(rows: list[dict]) -> bool:
    """Require a paired improvement on at least four of five seeds for one model."""
    for model in MODELS:
        deltas = []
        for row in rows:
            if row["model"] != model or row["kl_rl_validation_h"] == "":
                continue
            deltas.append(float(row["kl_rl_validation_h"]) - float(row["bc_validation_h"]))
        if len(deltas) < 5:
            continue
        mean = sum(deltas) / len(deltas)
        better = sum(delta < -1e-4 for delta in deltas)
        var = sum((delta - mean) ** 2 for delta in deltas) / (len(deltas) - 1)
        half = 2.7764451051977987 * math.sqrt(var / len(deltas))
        excludes_zero = mean + half < 0
        print({"model": model, "mean_delta_h": round(mean, 4), "better": better, "ci_excludes_zero": excludes_zero}, flush=True)
        if better >= 4 and excludes_zero:
            return True
    return False


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
                if (
                    int(meta.get("completed_rl_updates") or 0) >= 300
                    and float(meta.get("bc_kl") or 0) == 0.01
                    and meta.get("bc_kl_gradient") == "distribution"
                ):
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
    rows = write_validation_comparison()
    if validation_supports_official_eval(rows):
        sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
        from eval_bc_rl_official import evaluate_neural_construction

        for model in MODELS:
            for seed in SEEDS:
                run_dir = OUT / "runs" / f"{model}_seed{seed}"
                out_csv = OUT / "eval" / f"{model}_seed{seed}.csv"
                if not (run_dir / "rl_best.pt").exists():
                    continue
                if out_csv.exists() and out_csv.stat().st_size > 0:
                    continue
                print({"eval": f"{model}_seed{seed}", "mode": "neural-construction"}, flush=True)
                evaluate_neural_construction(run_dir, ROOT / "dataset", out_csv)
    else:
        print({"official_eval": "not started", "reason": "validation did not support a KL improvement"}, flush=True)
    print({"failures": failures}, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
