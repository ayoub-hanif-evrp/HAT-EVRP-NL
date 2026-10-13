"""Controlled BC-only vs RL fine-tuning for RQ4 (does not overwrite Cycle-3).

Protocol:
1) Train BC-only (400 steps, rl=0) → checkpoints/*_bc_only_rq4.pt
2) Warm-start RL (40 standard REINFORCE steps) from those BC weights
3) Evaluate synthetic validation + optional official n10 construction-only

Seeds: 42,43,44 for H2 and H4 by default.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "raw" / "reward_learning"
CKPT = ROOT / "checkpoints"
RES = ROOT / "results"


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def train_bc(model: str, seed: int) -> Path:
    version = "bc_only_rq4"
    ckpt = CKPT / f"{model}_n10_seed{seed}_{version}.pt"
    if ckpt.exists():
        print(f"exists {ckpt}", flush=True)
        return ckpt
    run(
        [
            sys.executable,
            "train.py",
            "--model",
            model,
            "--size",
            "10",
            "--steps",
            "400",
            "--rl-steps",
            "0",
            "--train-instances",
            "120",
            "--val-instances",
            "12",
            "--eval-every",
            "40",
            "--seed",
            str(seed),
            "--version",
            version,
            "--generator",
            "v2",
        ]
    )
    # Move training log into raw folder for archival clarity
    log = RES / f"training_{model}_n10_seed{seed}_{version}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    if log.exists():
        target = OUT / log.name
        log.replace(target)
    return ckpt


def train_rl_from_bc(model: str, seed: int, bc_ckpt: Path) -> Path:
    version = "rl_from_bc_rq4"
    ckpt = CKPT / f"{model}_n10_seed{seed}_{version}.pt"
    if ckpt.exists():
        print(f"exists {ckpt}", flush=True)
        return ckpt
    run(
        [
            sys.executable,
            "train.py",
            "--model",
            model,
            "--size",
            "10",
            "--steps",
            "0",
            "--rl-steps",
            "40",
            "--train-instances",
            "120",
            "--val-instances",
            "12",
            "--eval-every",
            "10",
            "--seed",
            str(seed),
            "--version",
            version,
            "--generator",
            "v2",
            "--init-checkpoint",
            str(bc_ckpt),
        ]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for prefix in ("training_", "reward_"):
        log = RES / f"{prefix}{model}_n10_seed{seed}_{version}.csv"
        if log.exists():
            log.replace(OUT / log.name)
    return ckpt


def evaluate_official(model: str, seed: int, ckpt: Path, tag: str) -> None:
    """Construction-only official n10 (no LS) for BC vs RL attribution."""
    out = OUT / f"official_n10_{tag}_{model}_seed{seed}.csv"
    if out.exists():
        return
    # Use evaluate compare with single method via python snippet for clarity
    code = f"""
import csv
from pathlib import Path
import torch
from data import instance_files, parse_instance
from train import decode, load_policy
ROOT = Path(r'{ROOT}')
policy = load_policy(r'{ckpt}')
rows = []
with torch.no_grad():
    for path in instance_files(ROOT/'dataset'):
        inst = parse_instance(path)
        if len(inst.customers) != 10:
            continue
        out = decode(policy, inst, sample=False, seed={seed}, cost_lambda=0.0)
        rows.append({{
            'instance': inst.name,
            'method': '{model}',
            'tag': '{tag}',
            'seed': {seed},
            'checkpoint': r'{ckpt.name}',
            'feasible': int(out['feasible']),
            'objective_h': out['objective'] if out['feasible'] else '',
            'n_routes': len(out['routes']) if out['feasible'] else '',
        }})
        print(inst.name, out['feasible'], out.get('objective'))
path = Path(r'{out}')
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('wrote', path)
"""
    run([sys.executable, "-c", code])


def summarize() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ("H2", "H4"):
        for seed in (42, 43, 44):
            for tag in ("bc_only_rq4", "rl_from_bc_rq4"):
                path = OUT / f"official_n10_{tag}_{model}_seed{seed}.csv"
                if not path.exists():
                    continue
                data = list(csv.DictReader(path.open(encoding="utf-8")))
                feas = [r for r in data if r["feasible"] == "1"]
                mean = (
                    sum(float(r["objective_h"]) for r in feas) / len(feas)
                    if feas
                    else float("nan")
                )
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "tag": tag,
                        "n": len(data),
                        "n_feasible": len(feas),
                        "mean_full_h": mean,
                    }
                )
    path = OUT / "bc_vs_rl_official_n10_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}", flush=True)

    # Reward study markdown
    lines = [
        "# Reward and Learning Study (RQ4)",
        "",
        "Controlled BC-only vs standard REINFORCE fine-tuning.",
        "Does **not** overwrite Cycle-3 checkpoints.",
        "",
        "## Protocol",
        "",
        "- Generator `v2`, n=10 synthetic, 120 train / 12 val (same family as Cycle-3).",
        "- BC-only: 400 steps, `rl_steps=0` → `*_bc_only_rq4.pt`.",
        "- RL: warm-start from BC-only, `steps=0`, `rl_steps=40`, standard REINFORCE",
        "  (greedy baseline, `R=-Z_full` or `-100`) → `*_rl_from_bc_rq4.pt` + `reward_*.csv`.",
        "- Official evaluation: greedy construction-only on 20 n=10 Montoya instances.",
        "",
        "## Reward definition",
        "",
        "- Feasible: `R = -Z_full` (`Z_full = travel + service + nonlinear charging`).",
        "- Infeasible: `R = -100`.",
        "- Baseline: greedy rollout reward; `L = -(R-b) Σ log π`.",
        "- `frvcpy` is non-differentiable; gradients use sampled routing log-probabilities only.",
        "",
        "## Official n=10 construction-only summary",
        "",
        "| Model | Seed | Tag | Feasible | Mean full h |",
        "|---|---:|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['seed']} | {r['tag']} | {r['n_feasible']}/{r['n']} | "
            f"{float(r['mean_full_h']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Historical Cycle-3 log analysis (not controlled checkpoints)",
            "",
            "See `results/raw/training_logs/derived/bc_vs_rl_aggregate_c3.csv`:",
            "RL endpoints improved best-BC validation on 1/3 H2 seeds and 0/3 H4 seeds.",
            "",
            "## Raw",
            "",
            f"- `{OUT.relative_to(ROOT).as_posix()}/`",
            "",
        ]
    )
    report = ROOT / "results" / "REWARD_AND_LEARNING_STUDY.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="H2,H4")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--stage",
        choices=("bc", "rl", "eval", "summarize", "all"),
        default="all",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    bc_paths = {}
    if args.stage in ("bc", "all"):
        for model in models:
            for seed in seeds:
                bc_paths[(model, seed)] = train_bc(model, seed)
    else:
        for model in models:
            for seed in seeds:
                bc_paths[(model, seed)] = CKPT / f"{model}_n10_seed{seed}_bc_only_rq4.pt"

    rl_paths = {}
    if args.stage in ("rl", "all"):
        for model in models:
            for seed in seeds:
                rl_paths[(model, seed)] = train_rl_from_bc(
                    model, seed, bc_paths[(model, seed)]
                )

    if args.stage in ("eval", "all"):
        for model in models:
            for seed in seeds:
                bc = CKPT / f"{model}_n10_seed{seed}_bc_only_rq4.pt"
                rl = CKPT / f"{model}_n10_seed{seed}_rl_from_bc_rq4.pt"
                if bc.exists():
                    evaluate_official(model, seed, bc, "bc_only_rq4")
                if rl.exists():
                    evaluate_official(model, seed, rl, "rl_from_bc_rq4")

    if args.stage in ("summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()
