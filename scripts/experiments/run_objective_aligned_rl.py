"""One H2 REINFORCE run aimed at the best of one greedy route and three samples.

The score-function estimator multiplies (R_best - b) by the sum of the sample
log-probabilities. The greedy route is detached, so it can serve as a candidate
and as the baseline, and its own parameter dependence is not differentiated.
This script does not retrain behavior cloning and does not write into earlier
result directories. Selection uses only the development split.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "raw" / "bc_rl_full_v1"
OUT = ROOT / "results" / "raw" / "final_autonomous_study" / "objective_aligned"
CACHE = SOURCE / "teacher_cache_n10_train500_val100.pkl"
SELECTION = ROOT / "results" / "raw" / "final_autonomous_study" / "splits" / "selection_dev.pkl"
SEEDS = (42, 43, 44, 45, 46)
PROTOCOL = {
    "name": "objective_aligned_h2_v1",
    "hypothesis": (
        "The score-function gradient of the best reward among one detached greedy "
        "construction and three samples lowers development-set penalized returned-route "
        "duration relative to the saved H2 behavior-cloning checkpoints."
    ),
    "model": "H2",
    "seeds": list(SEEDS),
    "rl_steps": 300,
    "rl_lr": 2e-5,
    "rollouts": 3,
    "baseline": "best_of_k",
    "selection": "anchored",
    "selection_split": "dev_v2",
    "selection_cache": "results/raw/final_autonomous_study/splits/selection_dev.pkl",
    "sample_seed0": 950000,
    "batch_size": 2,
    "entropy_coef": 0.0,
    "bc_kl": 0.0,
    "reward": "normalized",
    "estimator": "E[(R_best-b)*sum_k grad log pi(sample_k)]",
    "estimator_assumptions": [
        "R_best is the maximum normalized reward among the detached greedy decode and the three samples.",
        "b is the detached greedy reward. The advantage is nonnegative and is zero when greedy wins.",
        "The same advantage multiplies every sample log-probability. The winning sample is not the only term.",
        "The greedy decode is under torch.no_grad. The estimator omits the greedy policy's dependence on the parameters.",
        "The estimator is the score function for the sampling distribution with the greedy candidate held fixed.",
    ],
    "decision_rule": (
        "Promising only if at least 4 of 5 seeds have a negative RL-minus-BC penalized "
        "anchored duration on dev_v2 and the paired Student-t 95% interval lies entirely below zero. "
        "confirm_v2 and the shift splits are not used to choose the checkpoint or the learning rate."
    ),
    "comparison_control": "results/raw/bc_rl_final_v1 greedy-rollout runs use 4 samples plus 1 greedy decode per instance; this run uses 3 samples plus 1 greedy decode.",
}


def command(seed: int) -> list[str]:
    run_id = f"H2_seed{seed}"
    source = SOURCE / "runs" / run_id / "bc_best.pt"
    return [
        sys.executable,
        "-u",
        str(ROOT / "train.py"),
        "--pipeline",
        "primary",
        "--version",
        "bcrl_objective_v1",
        "--model",
        "H2",
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
        "3",
        "--baseline",
        "best_of_k",
        "--selection",
        "anchored",
        "--selection-cache",
        str(SELECTION),
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "frozen_protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved != PROTOCOL:
            raise SystemExit("objective-aligned protocol already exists and does not match")
    else:
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    if not SELECTION.exists():
        raise SystemExit("development selection cache is missing")
    failures = []
    for seed in SEEDS:
        print({"start": f"H2_seed{seed}"}, flush=True)
        completed = subprocess.run(command(seed), cwd=ROOT, env=os.environ.copy())
        if completed.returncode != 0:
            failures.append(seed)
            print({"failed": seed, "returncode": completed.returncode}, flush=True)
        else:
            print({"trained": f"H2_seed{seed}"}, flush=True)
    print({"failures": failures}, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
