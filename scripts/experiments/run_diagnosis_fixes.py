"""Implement root-cause diagnosis fixes and measure outcomes.

1. Verify v2b station-count + Montoya-like prior rebalance
2. Measure construction basins (pre-LS best/mean of N)
3. Train H2 vs H4 with fleet-counterfactual loss (matched)
4. Official n=10 with ruin as labeled search post-process
5. Write DIAGNOSIS_FIX_REPORT.md
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
CKPT = ROOT / "checkpoints"
DATA = ROOT / "dataset"
VERSION = "perf_v2b"


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


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(xs) if xs else float("nan")


def stage_verify_generator(n: int = 200, seed_start: int = 900000) -> dict:
    from data import generate_synthetic

    n_st = Counter()
    techs = Counter()
    depot_bias_proxy = []
    for i in range(n):
        inst = generate_synthetic(10, seed_start + i, generator_version="v2")
        n_st[len(inst.stations)] += 1
        techs["+".join(sorted(inst.curves.keys()))] += 1
        dx, dy = inst.nodes[inst.depot].x, inst.nodes[inst.depot].y
        dcd = [
            math.hypot(dx - inst.nodes[c].x, dy - inst.nodes[c].y)
            for c in inst.customers
        ]
        depot_bias_proxy.append(statistics.mean(dcd))
    summary = {
        "n": n,
        "station_count_hist": dict(n_st),
        "frac_3plus_stations": sum(v for k, v in n_st.items() if k >= 3) / n,
        "tech_hist": dict(techs),
        "mean_depot_customer_km": _mean(depot_bias_proxy),
        "has_3_stations": n_st.get(3, 0) > 0,
    }
    (RES / "DIAG_generator_v2b_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    assert summary["has_3_stations"], "station-count fix failed: still no 3-station instances"
    return summary


def stage_construction_basins(
    samples: int = 16,
    cost_lambda: float = 4.0,
) -> list[dict]:
    """Pre-LS construction quality: best/mean of N (where H2_new loses)."""
    import torch
    from data import instance_files, parse_instance
    from train import decode, load_policy

    policies = {
        "H2_c3": CKPT / "H2_n10_seed42_c3.pt",
        "H4_c3": CKPT / "H4_n10_seed42_c3.pt",
        "H2_v2st": CKPT / "H2_n10_seed42_perf_v2st.pt",
        "H2_v2b": CKPT / f"H2_n10_seed42_{VERSION}.pt",
        "H4_v2b": CKPT / f"H4_n10_seed42_{VERSION}.pt",
        "H2_v2b_fc": CKPT / f"H2_n10_seed42_{VERSION}_fc.pt",
        "H4_v2b_fc": CKPT / f"H4_n10_seed42_{VERSION}_fc.pt",
    }
    loaded = {}
    for tag, path in policies.items():
        if path.exists():
            loaded[tag] = load_policy(path)

    selected = [
        path
        for path in instance_files(DATA)
        if len(parse_instance(path).customers) == 10
    ]
    rows = []
    with torch.no_grad():
        for path in selected:
            instance = parse_instance(path)
            for tag, policy in loaded.items():
                objs = []
                n_routes = []
                for i in range(samples):
                    result = decode(
                        policy,
                        instance,
                        sample=i > 0,
                        seed=42 * 1000 + i,
                        cost_lambda=cost_lambda,
                        local_passes=0,
                        local_budget=0,
                    )
                    if result["feasible"]:
                        objs.append(result["objective"])
                        n_routes.append(len(result["routes"]))
                rows.append(
                    {
                        "instance": instance.name,
                        "layout": instance.name[:3],
                        "method": tag,
                        "feas_rate": len(objs) / samples,
                        "best_construct_h": min(objs) if objs else "",
                        "mean_construct_h": _mean(objs) if objs else "",
                        "median_construct_h": statistics.median(objs) if objs else "",
                        "mean_routes": _mean(n_routes) if n_routes else "",
                        "samples": samples,
                        "cost_lambda": cost_lambda,
                    }
                )
                print(
                    f"basin {tag} {instance.name}: "
                    f"best={rows[-1]['best_construct_h']} mean={rows[-1]['mean_construct_h']}"
                )
    _write_csv(RES / "DIAG_construction_basins_n10.csv", rows)

    # Aggregate by method
    by_m = defaultdict(list)
    by_ml = defaultdict(list)
    for r in rows:
        if r["mean_construct_h"] != "":
            by_m[r["method"]].append(float(r["mean_construct_h"]))
            by_ml[(r["method"], r["layout"])].append(float(r["mean_construct_h"]))
    summary = {
        "mean_construct_by_method": {m: _mean(v) for m, v in sorted(by_m.items())},
        "mean_construct_by_method_layout": {
            f"{m}|{lay}": _mean(v) for (m, lay), v in sorted(by_ml.items())
        },
    }
    (RES / "DIAG_construction_basins_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return rows


def stage_train(
    models: tuple[str, ...] = ("H2", "H4"),
    fleet_cf: bool = False,
    train_n: int = 100,
    val_n: int = 30,
    steps: int = 150,
    seed: int = 42,
) -> None:
    version = f"{VERSION}_fc" if fleet_cf else VERSION
    for model in models:
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--model",
            model,
            "--size",
            "10",
            "--steps",
            str(steps),
            "--rl-steps",
            "0",
            "--train-instances",
            str(train_n),
            "--val-instances",
            str(val_n),
            "--eval-every",
            "25",
            "--seed",
            str(seed),
            "--version",
            version,
            "--teacher",
            "strong",
            "--generator",
            "v2",
            "--checkpoints",
            str(CKPT),
            "--results",
            str(RES),
        ]
        if fleet_cf:
            cmd.append("--fleet-counterfactual")
        print("RUN", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT))


def stage_official(
    version: str,
    ruin: bool,
    seeds: tuple[int, ...] = (42,),
    samples: int = 8,
    cost_lambda: float = 4.0,
    tag: str = "",
) -> list[dict]:
    import torch
    from charging import check_solution, literature_compatible_objective_hours
    from data import instance_files, parse_instance
    from evaluate import _refine_best_once
    from train import decode, load_policy

    bks = {}
    with (RES / "literature_bks_froger2021.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bks[row["instance"]] = float(row["bks_h"])

    selected = [
        path
        for path in instance_files(DATA)
        if len(parse_instance(path).customers) == 10
    ]
    methods = []
    for seed in seeds:
        for model in ("H2", "H4"):
            for ver, label in (
                ("c3", f"{model}_c3"),
                (version, f"{model}_v2b"),
                (f"{version}_fc", f"{model}_v2b_fc"),
            ):
                path = CKPT / f"{model}_n10_seed{seed}_{ver}.pt"
                if path.exists():
                    methods.append((label, model, seed, path))

    rows = []
    ruin_it, ruin_bud = (6, 600) if ruin else (0, 0)
    with torch.no_grad():
        for label, model, seed, path in methods:
            policy = load_policy(path)
            for path_inst in selected:
                instance = parse_instance(path_inst)
                attempts = [
                    decode(
                        policy,
                        instance,
                        sample=i > 0,
                        seed=seed * 1000 + i,
                        cost_lambda=cost_lambda,
                        local_passes=0,
                        local_budget=0,
                    )
                    for i in range(samples)
                ]
                construct_t = sum(a["runtime"] for a in attempts)
                construct_o = sum(a["oracle_calls"] for a in attempts)
                feasible = [a for a in attempts if a["feasible"]]
                if not feasible:
                    rows.append(
                        {
                            "instance": instance.name,
                            "layout": instance.name[:3],
                            "method": label,
                            "seed": seed,
                            "ruin_search": int(ruin),
                            "feasible": 0,
                            "objective_h": "",
                            "objective_lit_h": "",
                            "bks_gap_lit_pct": "",
                            "runtime_s": construct_t,
                            "oracle_calls": construct_o,
                            "best_construct_h": "",
                            "mean_construct_h": "",
                        }
                    )
                    continue
                construct_objs = [a["objective"] for a in feasible]
                best = min(feasible, key=lambda a: a["objective"])
                t0 = time.perf_counter()
                routes, plans, _, refine_o, _ = _refine_best_once(
                    instance,
                    list(best["routes"]),
                    list(best["plans"]),
                    local_passes=3,
                    local_budget=3400 if ruin else 4000,
                    operators="strong",
                    ruin_iterations=ruin_it,
                    ruin_budget=ruin_bud,
                    ruin_seed=seed + 91,
                )
                refine_t = time.perf_counter() - t0
                valid, _ = check_solution(instance, plans)
                obj = sum(p.duration_hours for p in plans) if valid else float("inf")
                lit = (
                    literature_compatible_objective_hours(instance, obj)
                    if valid
                    else float("inf")
                )
                gap = ""
                if valid and instance.name in bks:
                    gap = round(100.0 * (lit - bks[instance.name]) / bks[instance.name], 4)
                rows.append(
                    {
                        "instance": instance.name,
                        "layout": instance.name[:3],
                        "method": label,
                        "seed": seed,
                        "ruin_search": int(ruin),
                        "feasible": int(valid),
                        "objective_h": obj if valid else "",
                        "objective_lit_h": lit if valid else "",
                        "bks_gap_lit_pct": gap,
                        "runtime_s": round(construct_t + refine_t, 4),
                        "oracle_calls": construct_o + refine_o,
                        "best_construct_h": min(construct_objs),
                        "mean_construct_h": _mean(construct_objs),
                    }
                )
                print(
                    f"official {label} ruin={ruin} {instance.name} "
                    f"obj={obj if valid else 'inf'} gap={gap}"
                )
    out = RES / f"DIAG_official_n10_{tag or ('ruin' if ruin else 'ls')}.csv"
    _write_csv(out, rows)
    return rows


def stage_report() -> None:
    gen = {}
    gp = RES / "DIAG_generator_v2b_summary.json"
    if gp.exists():
        gen = json.loads(gp.read_text(encoding="utf-8"))
    basins = {}
    bp = RES / "DIAG_construction_basins_summary.json"
    if bp.exists():
        basins = json.loads(bp.read_text(encoding="utf-8"))

    def summarize_official(path: Path) -> dict:
        if not path.exists():
            return {}
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        by = defaultdict(lambda: {"obj": [], "gap": [], "bc": [], "mc": [], "n": 0, "f": 0})
        for r in rows:
            m = r["method"]
            by[m]["n"] += 1
            if r["feasible"] == "1":
                by[m]["f"] += 1
                by[m]["obj"].append(float(r["objective_h"]))
                if r["bks_gap_lit_pct"] != "":
                    by[m]["gap"].append(float(r["bks_gap_lit_pct"]))
                if r.get("best_construct_h", "") != "":
                    by[m]["bc"].append(float(r["best_construct_h"]))
                if r.get("mean_construct_h", "") != "":
                    by[m]["mc"].append(float(r["mean_construct_h"]))
        return {
            m: {
                "feas_pct": 100 * d["f"] / max(1, d["n"]),
                "mean_full_h": _mean(d["obj"]),
                "mean_lit_gap_pct": _mean(d["gap"]),
                "mean_best_construct_h": _mean(d["bc"]),
                "mean_mean_construct_h": _mean(d["mc"]),
            }
            for m, d in sorted(by.items())
        }

    ls = summarize_official(RES / "DIAG_official_n10_ls.csv")
    ruin = summarize_official(RES / "DIAG_official_n10_ruin.csv")

    lines = [
        "# Diagnosis Fix Report",
        "",
        "Implements the root-cause analysis fix order: generator rebalance,",
        "construction-basin metrics, fleet-counterfactual imitation, and ruin as",
        "**search-only** post-process.",
        "",
        "## 1. Generator v2b (station count + Montoya-like priors)",
        "",
        f"- Station-count histogram: `{gen.get('station_count_hist')}`",
        f"- Fraction with ≥3 stations: {gen.get('frac_3plus_stations')}",
        f"- Tech histogram: `{gen.get('tech_hist')}`",
        f"- Mean depot–customer distance: {gen.get('mean_depot_customer_km')}",
        "",
        "Fix: `math.ceil` density mapping + explicit station_count prior; weighted",
        "`fast+slow` / `normal+slow`; offset depot bias. No official coordinates copied.",
        "",
        "## 2. Construction basins (pre-LS)",
        "",
        "Mean construction objective by method:",
        "",
    ]
    for m, v in (basins.get("mean_construct_by_method") or {}).items():
        lines.append(f"- {m}: {v:.4f} h")
    lines += [
        "",
        "Artifact: `DIAG_construction_basins_n10.csv`.",
        "",
        "## 3. Official n=10 after LS (architecture compare; no ruin)",
        "",
        "| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, d in ls.items():
        lines.append(
            f"| {m} | {d['feas_pct']:.1f} | {d['mean_full_h']:.4f} | "
            f"{d['mean_lit_gap_pct']:.3f} | {d['mean_mean_construct_h']:.4f} |"
        )
    lines += [
        "",
        "## 4. Official n=10 + ruin (**search**, not architecture)",
        "",
        "| Method | Feas% | Mean full h | Mean lit gap% |",
        "|---|---:|---:|---:|",
    ]
    for m, d in ruin.items():
        lines.append(
            f"| {m} | {d['feas_pct']:.1f} | {d['mean_full_h']:.4f} | "
            f"{d['mean_lit_gap_pct']:.3f} |"
        )
    lines += [
        "",
        "## Conclusions",
        "",
        "- Generator now produces 3-station and Montoya-like tech mass.",
        "- Construction-basin metrics are first-class (pre-LS).",
        "- Fleet-counterfactual loss is available for H4 matched tests; only claim",
        "  DeepSets gains if H4_v2b_fc beats H2_v2b_fc under identical settings.",
        "- Ruin improvements are labeled `ruin_search` / `fairls_ruin_search`.",
        "",
        "## Commands",
        "",
        "```bash",
        "python results/run_diagnosis_fixes.py --stage all",
        "```",
        "",
    ]
    out = RES / "DIAGNOSIS_FIX_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "verify",
            "train",
            "train_fc",
            "basins",
            "official_ls",
            "official_ruin",
            "report",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--train-n", type=int, default=100)
    parser.add_argument("--val-n", type=int, default=30)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    stages = (
        [
            "verify",
            "train",
            "train_fc",
            "basins",
            "official_ls",
            "official_ruin",
            "report",
        ]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        print(f"\n===== STAGE {stage} =====", flush=True)
        t0 = time.perf_counter()
        if stage == "verify":
            stage_verify_generator()
        elif stage == "train":
            stage_train(
                fleet_cf=False,
                train_n=args.train_n,
                val_n=args.val_n,
                steps=args.steps,
            )
        elif stage == "train_fc":
            stage_train(
                fleet_cf=True,
                train_n=args.train_n,
                val_n=args.val_n,
                steps=args.steps,
            )
        elif stage == "basins":
            stage_construction_basins()
        elif stage == "official_ls":
            stage_official(VERSION, ruin=False, tag="ls")
        elif stage == "official_ruin":
            stage_official(VERSION, ruin=True, tag="ruin")
        elif stage == "report":
            stage_report()
        print(f"===== STAGE {stage} done in {time.perf_counter()-t0:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()
