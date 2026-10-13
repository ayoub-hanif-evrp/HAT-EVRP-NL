"""Fair repair-fix pipeline: open-route neural repair, matched decode/search.

Architecture frozen. No official-instance training. Search labeled as search.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
CKPT = ROOT / "checkpoints"
DATA = ROOT / "dataset"
VERSION = "perf_repair_fix"
RUIN_SEED = 42 + 91
RUIN_ITERS = 6
RUIN_BUDGET = 600
NEURAL_ITERS = 6
COST_LAMBDA = 4.0
SAMPLES = 8


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


def stage_tests() -> None:
    cmd = [sys.executable, "-m", "unittest", "tests.test_repair_and_validation", "-v"]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def stage_train(
    models: tuple[str, ...] = ("H2", "H4"),
    train_n: int = 80,
    val_n: int = 24,
    bc_steps: int = 60,
    rl_steps: int = 60,
    repair_rl_steps: int = 50,
    pomo_k: int = 8,
    seed: int = 42,
) -> None:
    for model in models:
        init = CKPT / f"{model}_n10_seed42_perf_cont_v2c.pt"
        if not init.exists():
            init = CKPT / f"{model}_n10_seed42_perf_v2b_fc.pt"
        if not init.exists():
            raise FileNotFoundError(f"missing warm-start: {init}")
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
            str(COST_LAMBDA),
            "--init-checkpoint",
            str(init),
            "--checkpoints",
            str(CKPT),
            "--results",
            str(RES),
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT))


def _methods() -> list[tuple[str, Path]]:
    methods = []
    for model in ("H2", "H4"):
        for ver, label in (
            ("c3", f"{model}_c3"),
            ("perf_cont_v2c", f"{model}_cont"),
            (VERSION, f"{model}_repair"),
        ):
            path = CKPT / f"{model}_n10_seed42_{ver}.pt"
            if path.exists():
                methods.append((label, path))
    return methods


def _construct_attempts(policy, inst, protocol: str, samples: int):
    from train import decode

    attempts = []
    for i in range(samples):
        force = None
        if protocol == "forced":
            force = list(inst.customers)[i % len(inst.customers)]
        attempts.append(
            decode(
                policy,
                inst,
                sample=(protocol == "forced") or i > 0,
                seed=42 * 1000 + i,
                cost_lambda=COST_LAMBDA,
                force_first_customer=force,
            )
        )
    return attempts


def stage_construct(size: int = 10, samples: int = SAMPLES) -> list[dict]:
    """Ordinary vs forced multi-start for ALL checkpoints (matched)."""
    import torch
    from data import instance_files, parse_instance
    from train import load_policy

    selected = [
        p for p in instance_files(DATA) if len(parse_instance(p).customers) == size
    ]
    if size != 10:
        selected = selected[:20]
    rows = []
    with torch.no_grad():
        for label, path in _methods():
            policy = load_policy(path)
            for protocol in ("ordinary", "forced"):
                for path_inst in selected:
                    inst = parse_instance(path_inst)
                    attempts = _construct_attempts(policy, inst, protocol, samples)
                    feasible = [a for a in attempts if a["feasible"]]
                    objs = [a["objective"] for a in feasible]
                    rows.append(
                        {
                            "instance": inst.name,
                            "layout": inst.name[:3],
                            "method": label,
                            "protocol": protocol,
                            "size": size,
                            "feasible_frac": len(feasible) / samples,
                            "best_construct_h": min(objs) if objs else "",
                            "mean_construct_h": _mean(objs) if objs else "",
                            "runtime_s": sum(a["runtime"] for a in attempts),
                            "oracle_calls": sum(a["oracle_calls"] for a in attempts),
                            "n_samples": samples,
                        }
                    )
                    print(
                        f"construct n={size} {protocol} {label} {inst.name} "
                        f"best={rows[-1]['best_construct_h']}"
                    )
    _write_csv(RES / f"REPAIR_construct_n{size}.csv", rows)
    return rows


def stage_matched_search(size: int = 10, samples: int = SAMPLES) -> list[dict]:
    """Classical vs neural vs hybrid from identical starting constructions."""
    import torch
    from charging import check_solution, literature_compatible_objective_hours
    from data import instance_files, parse_instance
    from evaluate import _refine_best_once
    from train import hybrid_lns, load_policy, neural_lns

    bks = {}
    with (RES / "literature_bks_froger2021.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bks[row["instance"]] = float(row["bks_h"])
    selected = [
        p for p in instance_files(DATA) if len(parse_instance(p).customers) == size
    ]
    if size != 10:
        selected = selected[:20]
    rows = []
    modes = ("ls", "classical_ruin", "neural_lns", "hybrid_lns")
    with torch.no_grad():
        for label, path in _methods():
            policy = load_policy(path)
            for path_inst in selected:
                inst = parse_instance(path_inst)
                # Identical forced multi-start construction for every method.
                attempts = _construct_attempts(policy, inst, "forced", samples)
                construct_t = sum(a["runtime"] for a in attempts)
                construct_o = sum(a["oracle_calls"] for a in attempts)
                feasible = [a for a in attempts if a["feasible"]]
                if not feasible:
                    for mode in modes:
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
                start_routes = [tuple(r) for r in best["routes"]]
                start_plans = list(best["plans"])

                for mode in modes:
                    t0 = time.perf_counter()
                    oracle_extra = 0
                    if mode == "ls":
                        routes, plans, _, oracle_extra, _ = _refine_best_once(
                            inst,
                            list(start_routes),
                            list(start_plans),
                            local_passes=3,
                            local_budget=4000,
                            operators="strong",
                        )
                    elif mode == "classical_ruin":
                        routes, plans, _, oracle_extra, _ = _refine_best_once(
                            inst,
                            list(start_routes),
                            list(start_plans),
                            local_passes=3,
                            local_budget=4000,
                            operators="strong",
                            ruin_iterations=RUIN_ITERS,
                            ruin_budget=RUIN_BUDGET,
                            ruin_seed=RUIN_SEED,
                        )
                    elif mode == "neural_lns":
                        # Same LS warm-start then pure neural repair (matched seed).
                        routes, plans, _, oracle_extra, _ = _refine_best_once(
                            inst,
                            list(start_routes),
                            list(start_plans),
                            local_passes=3,
                            local_budget=4000,
                            operators="strong",
                        )
                        routes, plans, _, diag = neural_lns(
                            policy,
                            inst,
                            routes,
                            plans,
                            max_iterations=NEURAL_ITERS,
                            seed=RUIN_SEED,
                            cost_lambda=COST_LAMBDA,
                            seed_classical=False,
                        )
                    else:  # hybrid_lns
                        routes, plans, _, oracle_extra, _ = _refine_best_once(
                            inst,
                            list(start_routes),
                            list(start_plans),
                            local_passes=3,
                            local_budget=4000,
                            operators="strong",
                        )
                        routes, plans, _, diag = hybrid_lns(
                            policy,
                            inst,
                            routes,
                            plans,
                            max_iterations=NEURAL_ITERS,
                            seed=RUIN_SEED,
                            cost_lambda=COST_LAMBDA,
                            classical_iterations=RUIN_ITERS,
                            classical_budget=RUIN_BUDGET,
                        )
                    refine_t = time.perf_counter() - t0
                    valid, _ = check_solution(inst, plans)
                    obj = (
                        sum(p.duration_hours for p in plans) if valid else float("inf")
                    )
                    lit = (
                        literature_compatible_objective_hours(inst, obj)
                        if valid
                        else float("inf")
                    )
                    gap = ""
                    if valid and inst.name in bks:
                        gap = round(
                            100.0 * (lit - bks[inst.name]) / bks[inst.name], 4
                        )
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
                            "oracle_calls": construct_o + oracle_extra,
                        }
                    )
                    print(
                        f"matched n={size} {mode} {label} {inst.name} "
                        f"obj={obj if valid else 'inf'} gap={gap}"
                    )
    _write_csv(RES / f"REPAIR_matched_n{size}.csv", rows)
    return rows


def _summarize(path: Path, key_fields=("method", "mode")) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by = defaultdict(lambda: {"obj": [], "gap": [], "mc": [], "rt": [], "oc": [], "n": 0, "f": 0})
    for r in rows:
        if "mode" in key_fields:
            key = f"{r['method']}|{r['mode']}"
        elif "protocol" in key_fields:
            key = f"{r['method']}|{r['protocol']}"
        else:
            key = r["method"]
        by[key]["n"] += 1
        feas = r.get("feasible", r.get("feasible_frac", "0"))
        is_feas = feas in ("1", "1.0") or (
            isinstance(feas, str) and feas.replace(".", "", 1).isdigit() and float(feas) > 0.99
        )
        if r.get("feasible") == "1" or (
            "feasible_frac" in r and float(r["feasible_frac"]) > 0
        ):
            if r.get("objective_h", "") not in ("", None):
                by[key]["f"] += 1
                by[key]["obj"].append(float(r["objective_h"]))
            if r.get("bks_gap_lit_pct", "") not in ("", None):
                by[key]["gap"].append(float(r["bks_gap_lit_pct"]))
            if r.get("mean_construct_h", "") not in ("", None):
                by[key]["mc"].append(float(r["mean_construct_h"]))
            if r.get("best_construct_h", "") not in ("", None) and "objective_h" not in r:
                by[key]["mc"].append(float(r["best_construct_h"]))
            if r.get("runtime_s", "") not in ("", None):
                by[key]["rt"].append(float(r["runtime_s"]))
            if r.get("oracle_calls", "") not in ("", None):
                by[key]["oc"].append(float(r["oracle_calls"]))
            if "feasible_frac" in r and "objective_h" not in r:
                by[key]["f"] += 1
    return {
        k: {
            "feas_pct": 100 * d["f"] / max(1, d["n"]),
            "mean_full_h": _mean(d["obj"]),
            "mean_lit_gap_pct": _mean(d["gap"]),
            "mean_construct_h": _mean(d["mc"]),
            "mean_runtime_s": _mean(d["rt"]),
            "mean_oracle_calls": _mean(d["oc"]),
        }
        for k, d in sorted(by.items())
    }


def stage_report() -> None:
    lines = [
        "# Repair-Fix Report",
        "",
        "Fixes: open-route neural repair, matched multi-start for all checkpoints,",
        "validation with `cost_lambda=4`, competing_frac ≤ 1, depot/empty route guards.",
        "Architecture frozen. No official training. Search labeled as search.",
        "",
    ]
    for size in (10, 20):
        cpath = RES / f"REPAIR_construct_n{size}.csv"
        if cpath.exists():
            lines.append(f"## Construction protocols n={size} (matched)")
            lines.append("")
            lines.append(
                "| Method | Protocol | Mean best construct h | Mean runtime s | Mean oracle |"
            )
            lines.append("|---|---|---:|---:|---:|")
            rows = list(csv.DictReader(cpath.open(encoding="utf-8")))
            by = defaultdict(list)
            for r in rows:
                if r["best_construct_h"] != "":
                    by[(r["method"], r["protocol"])].append(
                        (
                            float(r["best_construct_h"]),
                            float(r["runtime_s"]),
                            float(r["oracle_calls"]),
                        )
                    )
            for (method, protocol), vals in sorted(by.items()):
                lines.append(
                    f"| {method} | {protocol} | {_mean([v[0] for v in vals]):.4f} | "
                    f"{_mean([v[1] for v in vals]):.2f} | {_mean([v[2] for v in vals]):.0f} |"
                )
            lines.append("")

        mpath = RES / f"REPAIR_matched_n{size}.csv"
        if mpath.exists():
            lines.append(f"## Matched search n={size} (identical starts/seeds/budgets)")
            lines.append("")
            lines.append(
                "| Method | Mode | Feas% | Mean full h | Mean lit gap% | Mean runtime |"
            )
            lines.append("|---|---|---:|---:|---:|---:|")
            summary = _summarize(mpath)
            for key, d in summary.items():
                method, mode = key.split("|", 1)
                lines.append(
                    f"| {method} | {mode} | {d['feas_pct']:.1f} | "
                    f"{d['mean_full_h']:.4f} | {d['mean_lit_gap_pct']:.3f} | "
                    f"{d['mean_runtime_s']:.2f} |"
                )
            lines.append("")

            # Paired classical vs neural vs hybrid for repair checkpoint
            rows = list(csv.DictReader(mpath.open(encoding="utf-8")))
            paired = defaultdict(dict)
            for r in rows:
                if r["feasible"] != "1" or r["bks_gap_lit_pct"] == "":
                    continue
                paired[(r["method"], r["instance"])][r["mode"]] = float(
                    r["bks_gap_lit_pct"]
                )
            deltas = defaultdict(list)
            for (method, _inst), modes in paired.items():
                if "classical_ruin" in modes and "neural_lns" in modes:
                    deltas[(method, "neural-classical")].append(
                        modes["neural_lns"] - modes["classical_ruin"]
                    )
                if "classical_ruin" in modes and "hybrid_lns" in modes:
                    deltas[(method, "hybrid-classical")].append(
                        modes["hybrid_lns"] - modes["classical_ruin"]
                    )
            lines.append(f"### Paired gap deltas n={size} (positive ⇒ classical better)")
            lines.append("")
            for (method, name), vals in sorted(deltas.items()):
                lines.append(
                    f"- {method} {name}: mean { _mean(vals):.4f} pp "
                    f"(n={len(vals)})"
                )
            lines.append("")

    # Headline conclusions
    best = (float("inf"), "")
    m10 = RES / "REPAIR_matched_n10.csv"
    if m10.exists():
        for key, d in _summarize(m10).items():
            gap = d["mean_lit_gap_pct"]
            if math.isfinite(gap) and gap < best[0]:
                best = (gap, key)
    lines.extend(
        [
            "## Conclusions",
            "",
            f"- Best matched n=10 stack: **{best[1]}** (mean lit gap {best[0]:.3f}%)",
            "- Neural repair now keeps surviving routes **open** for re-insertion.",
            "- Old vs new construction uses identical ordinary and forced protocols.",
            "- If neural/hybrid does not beat classical ruin under matched budgets, "
            "classical ruin remains the practical search default.",
            "",
            "## Commands",
            "",
            "```bash",
            "python results/run_repair_fix.py --stage all",
            "```",
            "",
        ]
    )
    out = RES / "REPAIR_FIX_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "tests",
            "train",
            "construct10",
            "construct20",
            "matched10",
            "matched20",
            "report",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--bc-steps", type=int, default=60)
    parser.add_argument("--rl-steps", type=int, default=60)
    parser.add_argument("--repair-rl-steps", type=int, default=50)
    args = parser.parse_args()
    t0 = time.perf_counter()
    if args.stage in ("tests", "all"):
        print("===== STAGE tests =====", flush=True)
        stage_tests()
    if args.stage in ("train", "all"):
        print("===== STAGE train =====", flush=True)
        stage_train(
            bc_steps=args.bc_steps,
            rl_steps=args.rl_steps,
            repair_rl_steps=args.repair_rl_steps,
        )
    if args.stage in ("construct10", "all"):
        print("===== STAGE construct10 =====", flush=True)
        stage_construct(10)
    if args.stage in ("matched10", "all"):
        print("===== STAGE matched10 =====", flush=True)
        stage_matched_search(10)
    if args.stage in ("construct20", "all"):
        print("===== STAGE construct20 =====", flush=True)
        stage_construct(20, samples=4)
    if args.stage in ("matched20", "all"):
        print("===== STAGE matched20 =====", flush=True)
        stage_matched_search(20, samples=4)
    if args.stage in ("report", "all"):
        print("===== STAGE report =====", flush=True)
        stage_report()
    print(f"===== done in {time.perf_counter() - t0:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()
