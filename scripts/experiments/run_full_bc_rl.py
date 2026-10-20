"""Run the predeclared 4-model x 5-seed BC→RL matrix.

Does not reduce the update counts. Skips a run only when its manifest row is
already complete and the three checkpoints exist. Exits nonzero if any
required run is missing or failed.
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
OUT = ROOT / "results" / "raw" / "bc_rl_full_v1"
MANIFEST = OUT / "experiment_manifest.csv"
CACHE = OUT / "teacher_cache_n10_train500_val100.pkl"
LOGS = OUT / "logs"

MODELS = ("H1", "H2", "H3", "H4")
SEEDS = (42, 43, 44, 45, 46)
VERSION = "bcrl_full_v1"
DATA_SEED = 1000

PROTOCOL = {
    "size": 10,
    "train_instances": 500,
    "val_instances": 100,
    "steps": 1000,
    "rl_steps": 300,
    "eval_every": 25,
    "bc_lr": 0.0005,
    "rl_lr": 0.00005,
    "batch_size": 2,
    "rollouts": 4,
    "baseline": "loo",
    "reward": "normalized",
    "generator": "v2",
    "data_seed": DATA_SEED,
}

FIELDS = [
    "model",
    "seed",
    "status",
    "dataset",
    "requested_bc_updates",
    "completed_bc_updates",
    "requested_rl_updates",
    "completed_rl_updates",
    "n_train",
    "n_val",
    "rollouts",
    "run_dir",
    "bc_checkpoint",
    "rl_checkpoint",
    "deployed_checkpoint",
    "training_log",
    "reward_log",
    "stdout_log",
    "eval_csv",
    "started_utc",
    "finished_utc",
    "wall_s",
    "failure",
]


def command(model: str, seed: int) -> list[str]:
    run_id = f"{model}_seed{seed}"
    return [
        sys.executable,
        "-u",
        str(ROOT / "train.py"),
        "--pipeline",
        "primary",
        "--version",
        VERSION,
        "--model",
        model,
        "--size",
        str(PROTOCOL["size"]),
        "--seed",
        str(seed),
        "--data-seed",
        str(DATA_SEED),
        "--generator",
        PROTOCOL["generator"],
        "--train-instances",
        str(PROTOCOL["train_instances"]),
        "--val-instances",
        str(PROTOCOL["val_instances"]),
        "--steps",
        str(PROTOCOL["steps"]),
        "--rl-steps",
        str(PROTOCOL["rl_steps"]),
        "--eval-every",
        str(PROTOCOL["eval_every"]),
        "--bc-lr",
        str(PROTOCOL["bc_lr"]),
        "--rl-lr",
        str(PROTOCOL["rl_lr"]),
        "--batch-size",
        str(PROTOCOL["batch_size"]),
        "--rollouts",
        str(PROTOCOL["rollouts"]),
        "--baseline",
        PROTOCOL["baseline"],
        "--reward",
        PROTOCOL["reward"],
        "--teacher-cache",
        str(CACHE),
        "--run-dir",
        str(OUT / "runs"),
        "--run-id",
        run_id,
    ]


def write_manifest(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def checkpoints_ok(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in ("bc_best.pt", "rl_best.pt", "deployed_best.pt", "run_meta.json", "training.csv"))


def training_record_ok(run_dir: Path) -> tuple[bool, str]:
    """Require update counts and checkpoint payloads, not only nonempty files."""
    if not checkpoints_ok(run_dir):
        return False, "missing checkpoint or log"
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if int(meta.get("completed_bc_updates") or 0) < PROTOCOL["steps"]:
        return False, f"bc updates {meta.get('completed_bc_updates')}"
    if int(meta.get("completed_rl_updates") or 0) < PROTOCOL["rl_steps"]:
        return False, f"rl updates {meta.get('completed_rl_updates')}"
    if int(meta.get("n_train") or 0) != PROTOCOL["train_instances"]:
        return False, "n_train mismatch"
    try:
        import torch

        for name in ("bc_best.pt", "rl_best.pt", "deployed_best.pt"):
            blob = torch.load(run_dir / name, map_location="cpu", weights_only=False)
            if "state_dict" not in blob or blob.get("model") != meta.get("model"):
                return False, f"{name} payload"
    except Exception as exc:
        return False, f"checkpoint read {exc}"
    return True, ""


def evaluation_record_ok(csv_path: Path) -> tuple[bool, str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False, "missing eval csv"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # 2 checkpoints x 4 sizes x 20 instances x 3 protocols.
    if len(rows) != 480:
        return False, f"eval rows {len(rows)}"
    p2 = [row for row in rows if row.get("protocol") == "P2"]
    if len(p2) != 160:
        return False, f"p2 rows {len(p2)}"
    if any(row.get("accounting") != "construction_plus_hybrid" for row in p2):
        return False, "p2 accounting"
    return True, ""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_protocol.json").write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    rows: list[dict] = []
    failures = 0
    for model in MODELS:
        for seed in SEEDS:
            run_id = f"{model}_seed{seed}"
            run_dir = OUT / "runs" / run_id
            row = {
                "model": model,
                "seed": seed,
                "status": "pending",
                "dataset": f"v2 data_seed={DATA_SEED} n=10 train=500 val=100",
                "requested_bc_updates": PROTOCOL["steps"],
                "requested_rl_updates": PROTOCOL["rl_steps"],
                "n_train": PROTOCOL["train_instances"],
                "n_val": PROTOCOL["val_instances"],
                "rollouts": PROTOCOL["rollouts"],
                "run_dir": str(run_dir),
                "bc_checkpoint": str(run_dir / "bc_best.pt"),
                "rl_checkpoint": str(run_dir / "rl_best.pt"),
                "deployed_checkpoint": str(run_dir / "deployed_best.pt"),
                "training_log": str(run_dir / "training.csv"),
                "reward_log": str(run_dir / "reward.csv"),
                "stdout_log": str(LOGS / f"{run_id}.log"),
                "eval_csv": str(OUT / "eval" / f"{run_id}.csv"),
            }
            if checkpoints_ok(run_dir):
                ok, reason = training_record_ok(run_dir)
                meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
                row.update(
                    {
                        "status": "trained" if ok else "record_invalid",
                        "completed_bc_updates": meta.get("completed_bc_updates", ""),
                        "completed_rl_updates": meta.get("completed_rl_updates", ""),
                        "failure": "" if ok else reason,
                    }
                )
                if not ok:
                    failures += 1
                print({"skip_complete": run_id, "ok": ok, "reason": reason}, flush=True)
                rows.append(row)
                write_manifest(rows)
                continue
            if run_dir.exists() and not checkpoints_ok(run_dir):
                parked = run_dir.with_name(run_dir.name + "_incomplete_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
                run_dir.rename(parked)
                row["failure"] = f"previous incomplete dir moved to {parked.name}"
            started = time.time()
            row["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
            row["status"] = "running"
            rows.append(row)
            write_manifest(rows)
            cmd = command(model, seed)
            (LOGS / f"{run_id}.cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
            log_path = LOGS / f"{run_id}.log"
            print({"start": run_id}, flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            finished = time.time()
            row["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished))
            row["wall_s"] = round(finished - started, 1)
            if proc.returncode != 0 or not checkpoints_ok(run_dir):
                row["status"] = "failed"
                row["failure"] = f"exit {proc.returncode}; checkpoints_ok={checkpoints_ok(run_dir)}"
                failures += 1
            else:
                meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
                row["status"] = "trained"
                row["completed_bc_updates"] = meta.get("completed_bc_updates", "")
                row["completed_rl_updates"] = meta.get("completed_rl_updates", "")
                if int(row["completed_bc_updates"] or 0) < PROTOCOL["steps"] or int(row["completed_rl_updates"] or 0) < PROTOCOL["rl_steps"]:
                    row["status"] = "incomplete"
                    row["failure"] = "stopped before the requested update counts (patience or crash)"
                    failures += 1
            write_manifest(rows)
            print({"done": run_id, "status": row["status"], "wall_s": row["wall_s"]}, flush=True)
    for row in rows:
        if row["status"] != "trained":
            continue
        run_id = f"{row['model']}_seed{row['seed']}"
        run_dir = Path(row["run_dir"])
        eval_csv = OUT / "eval" / f"{run_id}.csv"
        eval_ok, eval_reason = evaluation_record_ok(eval_csv)
        if eval_ok:
            row["status"] = "trained_and_evaluated"
            row["eval_csv"] = str(eval_csv)
            continue
        if eval_csv.exists():
            print({"eval_incomplete": run_id, "reason": eval_reason}, flush=True)
        eval_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "experiments" / "eval_bc_rl_official.py"),
            str(run_dir),
            str(ROOT / "dataset"),
            str(eval_csv),
        ]
        with (LOGS / f"{run_id}_eval.log").open("w", encoding="utf-8") as log:
            eval_proc = subprocess.run(eval_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        eval_ok, eval_reason = evaluation_record_ok(eval_csv)
        if eval_proc.returncode != 0 or not eval_ok:
            row["status"] = "trained_eval_failed"
            row["failure"] = f"training finished; official eval exit {eval_proc.returncode}; {eval_reason}"
            failures += 1
        else:
            row["status"] = "trained_and_evaluated"
            row["eval_csv"] = str(eval_csv)
        write_manifest(rows)
        print({"evaluated": run_id, "status": row["status"]}, flush=True)
    print({"failures": failures, "manifest": str(MANIFEST)}, flush=True)
    return 1 if failures or len(rows) != len(MODELS) * len(SEEDS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
