"""Continue close-gap: warm-start v2b_fc → v2c POMO + repair-RL → hybrid LNS.

Preserves architecture, bans official-instance training, labels search gains.
"""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
CKPT = ROOT / "checkpoints"
DATA = ROOT / "dataset"
VERSION = "perf_cont_v2c"


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


def stage_train(
    models: tuple[str, ...] = ("H2", "H4"),
    train_n: int = 80,
    val_n: int = 24,
    bc_steps: int = 60,
    rl_steps: int = 60,
    repair_rl_steps: int = 40,
    pomo_k: int = 8,
    seed: int = 42,
) -> None:
    for model in models:
        init = CKPT / f"{model}_n10_seed42_perf_v2b_fc.pt"
        if not init.exists():
            raise FileNotFoundError(f"missing warm-start checkpoint: {init}")
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
            "--repair-rl-steps",
            str(repair_rl_steps),
            "--repair-k",
            "4",
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
            "--init-checkpoint",
            str(init),
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
        ("H2_v2b_fc", CKPT / "H2_n10_seed42_perf_v2b_fc.pt"),
        ("H4_v2b_fc", CKPT / "H4_n10_seed42_perf_v2b_fc.pt"),
        ("H2_cont", CKPT / f"H2_n10_seed42_{VERSION}.pt"),
        ("H4_cont", CKPT / f"H4_n10_seed42_{VERSION}.pt"),
    ):
        if path.exists():
            policies[tag] = load_policy(path)
    selected = [
        p for p in instance_files(DATA) if len(parse_instance(p).customers) == 10
    ]
    rows = []
    with torch.no_grad():
        for tag, policy in policies.items():
            for path_inst in selected:
                inst = parse_instance(path_inst)
                objs = []
                for i in range(samples):
                    force = list(inst.customers)[i % len(inst.customers)]
                    out = decode(
                        policy,
                        inst,
                        sample=True,
                        seed=42 * 1000 + i,
                        cost_lambda=cost_lambda,
                        force_first_customer=force,
                    )
                    if out["feasible"]:
                        objs.append(out["objective"])
                rows.append(
                    {
                        "instance": inst.name,
                        "method": tag,
                        "best_construct_h": min(objs) if objs else "",
                        "mean_construct_h": _mean(objs) if objs else "",
                        "feas_frac": len(objs) / samples,
                    }
                )
                print(
                    f"basin {tag} {inst.name} "
                    f"mean={rows[-1]['mean_construct_h']} best={rows[-1]['best_construct_h']}"
                )
    _write_csv(RES / "CONT_basins_n10.csv", rows)
    by = defaultdict(list)
    for r in rows:
        if r["mean_construct_h"] != "":
            by[r["method"]].append(float(r["mean_construct_h"]))
    summary = {m: _mean(v) for m, v in sorted(by.items())}
    (RES / "CONT_basins_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return rows


def stage_official(size: int = 10, mode: str = "ls", samples: int = 8) -> list[dict]:
    import torch
    from charging import check_solution, literature_compatible_objective_hours
    from data import instance_files, parse_instance
    from evaluate import _refine_best_once
    from train import decode, hybrid_lns, load_policy, neural_lns

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
            ("perf_v2b_fc", f"{model}_v2b_fc"),
            (VERSION, f"{model}_cont"),
        ):
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
                    # Matched multi-start for ALL checkpoints (not only continued).
                    force = list(inst.customers)[i % len(inst.customers)]
                    attempts.append(
                        decode(
                            policy,
                            inst,
                            sample=True,
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
                if mode == "hybrid_lns":
                    routes, plans, _, _diag = hybrid_lns(
                        policy,
                        inst,
                        list(best["routes"]),
                        list(best["plans"]),
                        max_iterations=5,
                        seed=42 + 7,
                        cost_lambda=4.0,
                        classical_iterations=6,
                        classical_budget=600,
                    )
                    # Match ruin-mode finishing LS budget so search is comparable.
                    routes, plans, _, refine_o, _ = _refine_best_once(
                        inst,
                        routes,
                        plans,
                        local_passes=3,
                        local_budget=4000,
                        operators="strong",
                    )
                elif mode == "neural_lns":
                    routes, plans, _, _diag = neural_lns(
                        policy,
                        inst,
                        list(best["routes"]),
                        list(best["plans"]),
                        max_iterations=5,
                        seed=42 + 7,
                        cost_lambda=4.0,
                    )
                    routes, plans, _, refine_o, _ = _refine_best_once(
                        inst,
                        routes,
                        plans,
                        local_passes=3,
                        local_budget=4000,
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
    out = RES / f"CONT_official_n{size}_{mode}.csv"
    _write_csv(out, rows)
    return rows


def _summarize(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
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


def _table(title: str, summary: dict) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, d in summary.items():
        lines.append(
            f"| {method} | {d['feas_pct']:.1f} | {d['mean_full_h']:.4f} | "
            f"{d['mean_lit_gap_pct']:.3f} | {d['mean_construct_h']:.4f} |"
        )
    lines.append("")
    return lines


def stage_report() -> None:
    basins = {}
    bp = RES / "CONT_basins_summary.json"
    if bp.exists():
        basins = json.loads(bp.read_text(encoding="utf-8"))
    lines = [
        "# Continue Report",
        "",
        "Warm-start from `perf_v2b_fc` → short BC on v2c + fleet-CF → POMO →",
        "repair-RL → hybrid LNS (classical ruin floor + neural attempts).",
        "Architecture frozen. No official-instance training. Search labeled.",
        "",
        "## Construction basins (pre-LS)",
        "",
    ]
    for method, value in sorted(basins.items()):
        lines.append(f"- {method}: mean construct {value:.4f} h")
    lines.append("")
    for size, mode, title in (
        (10, "ls", "n=10 + strong LS (architecture)"),
        (10, "ruin", "n=10 + classical ruin (**search**)"),
        (10, "hybrid_lns", "n=10 + hybrid LNS (**search**)"),
        (20, "ls", "n=20 + strong LS"),
        (20, "ruin", "n=20 + classical ruin (**search**)"),
        (20, "hybrid_lns", "n=20 + hybrid LNS (**search**)"),
    ):
        summary = _summarize(RES / f"CONT_official_n{size}_{mode}.csv")
        if summary:
            lines.extend(_table(title, summary))

    # Best practical headline
    best_gap = float("inf")
    best_name = ""
    for size, mode in ((10, "ruin"), (10, "hybrid_lns")):
        for method, d in _summarize(RES / f"CONT_official_n{size}_{mode}.csv").items():
            gap = d["mean_lit_gap_pct"]
            if math.isfinite(gap) and gap < best_gap:
                best_gap = gap
                best_name = f"{method} + {mode}"
    h2 = _summarize(RES / "CONT_official_n10_hybrid_lns.csv").get("H2_cont", {})
    h4 = _summarize(RES / "CONT_official_n10_hybrid_lns.csv").get("H4_cont", {})
    lines.extend(
        [
            "## Conclusions",
            "",
            f"- Best n=10 search stack in this run: **{best_name}** "
            f"(mean lit gap {best_gap:.3f}%)",
            f"- H2_cont hybrid gap: {h2.get('mean_lit_gap_pct', float('nan'))}",
            f"- H4_cont hybrid gap: {h4.get('mean_lit_gap_pct', float('nan'))}",
            "- Hybrid LNS must not be worse than classical ruin (floor guarantee).",
            "- H4 > H2 only under matched settings with layout CI excluding 0.",
            "",
            "## Commands",
            "",
            "```bash",
            "python results/run_continue.py --stage all",
            "```",
            "",
        ]
    )
    out = RES / "CONTINUE_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "train",
            "basins",
            "official10",
            "official20",
            "report",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--bc-steps", type=int, default=60)
    parser.add_argument("--rl-steps", type=int, default=60)
    parser.add_argument("--repair-rl-steps", type=int, default=40)
    parser.add_argument("--pomo-k", type=int, default=8)
    args = parser.parse_args()
    t0 = time.perf_counter()
    if args.stage in ("train", "all"):
        print("===== STAGE train =====", flush=True)
        stage_train(
            bc_steps=args.bc_steps,
            rl_steps=args.rl_steps,
            repair_rl_steps=args.repair_rl_steps,
            pomo_k=args.pomo_k,
        )
    if args.stage in ("basins", "all"):
        print("===== STAGE basins =====", flush=True)
        stage_basins()
    if args.stage in ("official10", "all"):
        print("===== STAGE official10 =====", flush=True)
        stage_official(10, "ls")
        stage_official(10, "ruin")
        stage_official(10, "hybrid_lns")
    if args.stage in ("official20", "all"):
        print("===== STAGE official20 =====", flush=True)
        stage_official(20, "ls", samples=4)
        stage_official(20, "ruin", samples=4)
        stage_official(20, "hybrid_lns", samples=4)
    if args.stage in ("report", "all"):
        print("===== STAGE report =====", flush=True)
        stage_report()
    print(f"===== done in {time.perf_counter() - t0:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()
