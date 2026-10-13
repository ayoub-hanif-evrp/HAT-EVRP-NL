"""Close-gap pipeline: v2c gate → BC+fleet-CF → POMO RL → neural LNS eval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
CKPT = ROOT / "checkpoints"
DATA = ROOT / "dataset"
VERSION = "perf_v2c_pomo"


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(xs) if xs else float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def stage_gate() -> dict:
    from data import assert_distribution_gate, distribution_gate_metrics

    metrics = distribution_gate_metrics(n=500, generator_version="v2c")
    metrics = assert_distribution_gate(metrics)
    (RES / "CLOSE_v2c_gate.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def stage_train(
    models: tuple[str, ...] = ("H2", "H4"),
    train_n: int = 80,
    val_n: int = 24,
    bc_steps: int = 100,
    rl_steps: int = 40,
    pomo_k: int = 8,
    seed: int = 42,
) -> None:
    for model in models:
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--model",
            model,
            "--size",
            "10",
            "--steps",
            str(bc_steps),
            "--rl-steps",
            str(rl_steps),
            "--train-instances",
            str(train_n),
            "--val-instances",
            str(val_n),
            "--eval-every",
            "20",
            "--seed",
            str(seed),
            "--version",
            VERSION,
            "--teacher",
            "strong",
            "--generator",
            "v2c",
            "--fleet-counterfactual",
            "--pomo-k",
            str(pomo_k),
            "--cost-lambda",
            "4",
            "--checkpoints",
            str(CKPT),
            "--results",
            str(RES),
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT))


def stage_basins(samples: int = 12, cost_lambda: float = 4.0) -> list[dict]:
    import torch
    from data import instance_files, parse_instance
    from train import decode, load_policy

    policies = {}
    for tag, path in (
        ("H2_c3", CKPT / "H2_n10_seed42_c3.pt"),
        ("H4_c3", CKPT / "H4_n10_seed42_c3.pt"),
        ("H2_v2c_pomo", CKPT / f"H2_n10_seed42_{VERSION}.pt"),
        ("H4_v2c_pomo", CKPT / f"H4_n10_seed42_{VERSION}.pt"),
        ("H2_v2b_fc", CKPT / "H2_n10_seed42_perf_v2b_fc.pt"),
        ("H4_v2b_fc", CKPT / "H4_n10_seed42_perf_v2b_fc.pt"),
    ):
        if path.exists():
            policies[tag] = load_policy(path)
    selected = [
        p for p in instance_files(DATA) if len(parse_instance(p).customers) == 10
    ]
    rows = []
    with torch.no_grad():
        for path in selected:
            inst = parse_instance(path)
            for tag, policy in policies.items():
                objs = []
                for i in range(samples):
                    # POMO-style first-customer diversity at inference for new models.
                    force = (
                        list(inst.customers)[i % len(inst.customers)]
                        if "pomo" in tag
                        else None
                    )
                    r = decode(
                        policy,
                        inst,
                        sample=i > 0 or force is not None,
                        seed=42 * 1000 + i,
                        cost_lambda=cost_lambda,
                        force_first_customer=force,
                    )
                    if r["feasible"]:
                        objs.append(r["objective"])
                rows.append(
                    {
                        "instance": inst.name,
                        "layout": inst.name[:3],
                        "method": tag,
                        "best_construct_h": min(objs) if objs else "",
                        "mean_construct_h": _mean(objs) if objs else "",
                        "feas_rate": len(objs) / samples,
                    }
                )
                print(
                    f"basin {tag} {inst.name}: best={rows[-1]['best_construct_h']} "
                    f"mean={rows[-1]['mean_construct_h']}"
                )
    _write_csv(RES / "CLOSE_basins_n10.csv", rows)
    by = defaultdict(list)
    for r in rows:
        if r["mean_construct_h"] != "":
            by[r["method"]].append(float(r["mean_construct_h"]))
    summary = {m: _mean(v) for m, v in sorted(by.items())}
    (RES / "CLOSE_basins_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return rows


def stage_official(size: int = 10, mode: str = "ls", samples: int = 8) -> list[dict]:
    import torch
    from charging import check_solution, literature_compatible_objective_hours
    from data import instance_files, parse_instance
    from evaluate import _refine_best_once
    from train import decode, load_policy, neural_lns

    bks = {}
    with (RES / "literature_bks_froger2021.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bks[row["instance"]] = float(row["bks_h"])
    selected = [
        p for p in instance_files(DATA) if len(parse_instance(p).customers) == size
    ]
    if size != 10:
        selected = selected[:20]
    methods = []
    for model in ("H2", "H4"):
        for ver, label in (
            ("c3", f"{model}_c3"),
            (VERSION, f"{model}_v2c_pomo"),
            ("perf_v2b_fc", f"{model}_v2b_fc"),
        ):
            # Prefer n10 checkpoints even for n20 transfer.
            path = CKPT / f"{model}_n10_seed42_{ver}.pt"
            if path.exists():
                methods.append((label, path))
    rows = []
    with torch.no_grad():
        for label, path in methods:
            policy = load_policy(path)
            for path_inst in selected:
                inst = parse_instance(path_inst)
                attempts = []
                for i in range(samples):
                    force = (
                        list(inst.customers)[i % len(inst.customers)]
                        if "pomo" in label
                        else None
                    )
                    attempts.append(
                        decode(
                            policy,
                            inst,
                            sample=i > 0 or force is not None,
                            seed=42 * 1000 + i,
                            cost_lambda=4.0,
                            force_first_customer=force,
                        )
                    )
                construct_t = sum(a["runtime"] for a in attempts)
                construct_o = sum(a["oracle_calls"] for a in attempts)
                feasible = [a for a in attempts if a["feasible"]]
                if not feasible:
                    rows.append(
                        {
                            "instance": inst.name,
                            "layout": inst.name[:3],
                            "method": label,
                            "mode": mode,
                            "size": size,
                            "feasible": 0,
                            "objective_h": "",
                            "bks_gap_lit_pct": "",
                            "best_construct_h": "",
                            "mean_construct_h": "",
                            "runtime_s": construct_t,
                            "oracle_calls": construct_o,
                        }
                    )
                    continue
                construct_objs = [a["objective"] for a in feasible]
                best = min(feasible, key=lambda a: a["objective"])
                t0 = time.perf_counter()
                if mode == "neural_lns":
                    routes, plans, _, diag = neural_lns(
                        policy,
                        inst,
                        list(best["routes"]),
                        list(best["plans"]),
                        max_iterations=5,
                        seed=42 + 7,
                        cost_lambda=4.0,
                    )
                    # Matched finishing LS budget.
                    routes, plans, _, refine_o, _ = _refine_best_once(
                        inst,
                        routes,
                        plans,
                        local_passes=2,
                        local_budget=2000,
                        operators="strong",
                    )
                else:
                    routes, plans, _, refine_o, _ = _refine_best_once(
                        inst,
                        list(best["routes"]),
                        list(best["plans"]),
                        local_passes=3,
                        local_budget=4000,
                        operators="strong",
                        ruin_iterations=6 if mode == "ruin" else 0,
                        ruin_budget=600 if mode == "ruin" else 0,
                        ruin_seed=42 + 91,
                    )
                refine_t = time.perf_counter() - t0
                valid, _ = check_solution(inst, plans)
                obj = sum(p.duration_hours for p in plans) if valid else float("inf")
                lit = (
                    literature_compatible_objective_hours(inst, obj)
                    if valid
                    else float("inf")
                )
                gap = ""
                if valid and inst.name in bks:
                    gap = round(100.0 * (lit - bks[inst.name]) / bks[inst.name], 4)
                rows.append(
                    {
                        "instance": inst.name,
                        "layout": inst.name[:3],
                        "method": label,
                        "mode": mode,
                        "size": size,
                        "feasible": int(valid),
                        "objective_h": obj if valid else "",
                        "bks_gap_lit_pct": gap,
                        "best_construct_h": min(construct_objs),
                        "mean_construct_h": _mean(construct_objs),
                        "runtime_s": round(construct_t + refine_t, 4),
                        "oracle_calls": construct_o + refine_o,
                    }
                )
                print(
                    f"official n={size} {mode} {label} {inst.name} "
                    f"obj={obj if valid else 'inf'} gap={gap}"
                )
    out = RES / f"CLOSE_official_n{size}_{mode}.csv"
    _write_csv(out, rows)
    return rows


def stage_report() -> None:
    gate = {}
    gp = RES / "CLOSE_v2c_gate.json"
    if gp.exists():
        gate = json.loads(gp.read_text(encoding="utf-8"))
    basins = {}
    bp = RES / "CLOSE_basins_summary.json"
    if bp.exists():
        basins = json.loads(bp.read_text(encoding="utf-8"))

    def summarize(path: Path) -> dict:
        if not path.exists():
            return {}
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        by = defaultdict(lambda: {"obj": [], "gap": [], "mc": [], "n": 0, "f": 0})
        for r in rows:
            m = r["method"]
            by[m]["n"] += 1
            if r["feasible"] == "1":
                by[m]["f"] += 1
                by[m]["obj"].append(float(r["objective_h"]))
                if r["bks_gap_lit_pct"] != "":
                    by[m]["gap"].append(float(r["bks_gap_lit_pct"]))
                if r.get("mean_construct_h", "") not in ("", None):
                    by[m]["mc"].append(float(r["mean_construct_h"]))
        return {
            m: {
                "feas_pct": 100 * d["f"] / max(1, d["n"]),
                "mean_full_h": _mean(d["obj"]),
                "mean_lit_gap_pct": _mean(d["gap"]),
                "mean_construct_h": _mean(d["mc"]),
            }
            for m, d in sorted(by.items())
        }

    lines = [
        "# Close-Gap Report",
        "",
        "Literature-backed pipeline: Montoya-moment v2c → fleet-CF BC → POMO RL →",
        "neural-LNS search. Architecture frozen. No official-instance training.",
        "",
        "## Phase A — v2c distribution gate",
        "",
        f"- mean depot–customer km: {gate.get('mean_depot_customer_km')}",
        f"- mean customer–customer km: {gate.get('mean_customer_customer_km')}",
        f"- frac ≥3 stations: {gate.get('frac_3plus_stations')}",
        f"- gate passed: {gate.get('gate_passed')}",
        "",
        "## Phase B — Construction basins (pre-LS)",
        "",
    ]
    for m, v in basins.items():
        lines.append(f"- {m}: mean construct {v:.4f} h")
    for size, mode, title in (
        (10, "ls", "n=10 + strong LS (architecture)"),
        (10, "ruin", "n=10 + classical ruin (**search**)"),
        (10, "neural_lns", "n=10 + neural LNS (**search**)"),
        (20, "ls", "n=20 + strong LS"),
        (20, "neural_lns", "n=20 + neural LNS (**search**)"),
    ):
        summary = summarize(RES / f"CLOSE_official_n{size}_{mode}.csv")
        if not summary:
            continue
        lines += [
            "",
            f"## {title}",
            "",
            "| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |",
            "|---|---:|---:|---:|---:|",
        ]
        for m, d in summary.items():
            lines.append(
                f"| {m} | {d['feas_pct']:.1f} | {d['mean_full_h']:.4f} | "
                f"{d['mean_lit_gap_pct']:.3f} | {d['mean_construct_h']:.4f} |"
            )
    lines += [
        "",
        "## Claims discipline",
        "",
        "- Neural LNS / ruin improvements are **search**, not DeepSets architecture.",
        "- H4 > H2 only if matched v2c+fleet-CF+POMO+same search and layout CI excludes 0.",
        "",
        "## Commands",
        "",
        "```bash",
        "python results/run_close_gap.py --stage all",
        "```",
        "",
    ]
    out = RES / "CLOSE_GAP_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "gate",
            "train",
            "basins",
            "official_ls",
            "official_ruin",
            "official_lns",
            "official20",
            "report",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--train-n", type=int, default=80)
    parser.add_argument("--val-n", type=int, default=24)
    parser.add_argument("--bc-steps", type=int, default=100)
    parser.add_argument("--rl-steps", type=int, default=40)
    parser.add_argument("--pomo-k", type=int, default=8)
    args = parser.parse_args()
    stages = (
        [
            "gate",
            "train",
            "basins",
            "official_ls",
            "official_ruin",
            "official_lns",
            "official20",
            "report",
        ]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        print(f"\n===== STAGE {stage} =====", flush=True)
        t0 = time.perf_counter()
        if stage == "gate":
            stage_gate()
        elif stage == "train":
            stage_train(
                train_n=args.train_n,
                val_n=args.val_n,
                bc_steps=args.bc_steps,
                rl_steps=args.rl_steps,
                pomo_k=args.pomo_k,
            )
        elif stage == "basins":
            stage_basins()
        elif stage == "official_ls":
            stage_official(10, "ls")
        elif stage == "official_ruin":
            stage_official(10, "ruin")
        elif stage == "official_lns":
            stage_official(10, "neural_lns")
        elif stage == "official20":
            stage_official(20, "ls", samples=4)
            stage_official(20, "neural_lns", samples=4)
        elif stage == "report":
            stage_report()
        print(f"===== STAGE {stage} done in {time.perf_counter()-t0:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()
