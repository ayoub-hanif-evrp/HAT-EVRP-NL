"""Position-aware neural repair experiments (does not overwrite prior CSVs)."""

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
COST_LAMBDA = 4.0
RUIN_SEED = 42 + 91
RUIN_ITERS = 6
RUIN_BUDGET = 600
NEURAL_ITERS = 6
PREFIX = "INSERTION"


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(xs) if xs else float("nan")


def _median(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.median(xs) if xs else float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def stage_tests() -> None:
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_insertion_repair",
        "tests.test_repair_and_validation",
        "-v",
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def _methods() -> list[tuple[str, Path]]:
    out = []
    for model in ("H2", "H4"):
        for ver, label in (
            ("c3", f"{model}_c3"),
            ("perf_cont_v2c", f"{model}_cont"),
            ("perf_repair_fix", f"{model}_repair"),
            ("perf_insert_v1", f"{model}_insert"),
        ):
            path = CKPT / f"{model}_n10_seed42_{ver}.pt"
            if path.exists():
                out.append((label, path))
    return out


def _instances(size: int, synthetic: bool = False, limit: int | None = None):
    from data import generate_synthetic, instance_files, parse_instance

    if synthetic:
        return [
            generate_synthetic(size, 880000 + i, generator_version="v2c")
            for i in range(limit or 12)
        ]
    selected = [
        parse_instance(p)
        for p in instance_files(DATA)
        if len(parse_instance(p).customers) == size
    ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _construct(policy, inst, samples: int, seed0: int = 42000):
    from train import decode

    attempts = []
    for i in range(samples):
        force = list(inst.customers)[i % len(inst.customers)]
        attempts.append(
            decode(
                policy,
                inst,
                sample=True,
                seed=seed0 + i,
                cost_lambda=COST_LAMBDA,
                force_first_customer=force,
            )
        )
    feasible = [a for a in attempts if a["feasible"]]
    construct_t = sum(a["runtime"] for a in attempts)
    construct_exec = sum(a.get("oracle_solver_executions", a["oracle_calls"]) for a in attempts)
    construct_req = sum(a.get("oracle_requests", a["oracle_calls"]) for a in attempts)
    return attempts, feasible, construct_t, construct_exec, construct_req


def _run_modes(policy, inst, start_routes, start_plans, modes: tuple[str, ...]):
    """Identical start → A/B/C/D/E search modes with shared accounting."""
    from charging import FixedRouteOracle, check_solution
    from evaluate import _refine_best_once
    from train import hybrid_lns, neural_lns

    rows = []
    for mode in modes:
        oracle = FixedRouteOracle(inst)
        t0 = time.perf_counter()
        routes = [tuple(r) for r in start_routes]
        plans = list(start_plans)
        diag = {}
        # Always warm-start with strong LS on a shared oracle where possible.
        # _refine_best_once creates its own oracle; we measure via returned calls
        # then continue neural work on a fresh shared oracle and sum executions.
        if mode == "ls":
            routes, plans, _, ls_calls, _ = _refine_best_once(
                inst,
                routes,
                plans,
                local_passes=3,
                local_budget=4000,
                operators="strong",
            )
            exec_calls = ls_calls
        elif mode == "classical_ruin":
            routes, plans, _, ls_calls, _ = _refine_best_once(
                inst,
                routes,
                plans,
                local_passes=3,
                local_budget=4000,
                operators="strong",
                ruin_iterations=RUIN_ITERS,
                ruin_budget=RUIN_BUDGET,
                ruin_seed=RUIN_SEED,
            )
            exec_calls = ls_calls
        elif mode in ("neural_append", "neural_insert"):
            routes, plans, _, ls_calls, _ = _refine_best_once(
                inst,
                routes,
                plans,
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
                repair_insertion=(mode == "neural_insert"),
                oracle=oracle,
            )
            exec_calls = ls_calls + oracle.calls
        elif mode == "hybrid_insert":
            routes, plans, _, ls_calls, _ = _refine_best_once(
                inst,
                routes,
                plans,
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
                repair_insertion=True,
                oracle=oracle,
            )
            exec_calls = ls_calls + oracle.calls
        else:
            raise ValueError(mode)
        runtime = time.perf_counter() - t0
        valid, _ = check_solution(inst, plans)
        obj = sum(p.duration_hours for p in plans) if valid else float("inf")
        rows.append(
            {
                "routes": routes,
                "plans": plans,
                "feasible": valid,
                "objective_h": obj,
                "runtime_s": runtime,
                "oracle_solver_executions": exec_calls,
                "oracle_requests": ls_calls + oracle.requests
                if mode not in ("ls", "classical_ruin")
                else ls_calls,
                "diag": diag,
            }
        )
    return rows


def stage_eval(
    size: int,
    *,
    synthetic: bool = False,
    samples: int = 6,
    limit: int | None = None,
    tag: str | None = None,
) -> list[dict]:
    import torch
    from charging import literature_compatible_objective_hours
    from train import load_policy

    bks = {}
    bks_path = RES / "literature_bks_froger2021.csv"
    if bks_path.exists() and not synthetic:
        with bks_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                bks[row["instance"]] = float(row["bks_h"])

    modes = (
        "ls",
        "classical_ruin",
        "neural_append",
        "neural_insert",
        "hybrid_insert",
    )
    instances = _instances(size, synthetic=synthetic, limit=limit)
    rows = []
    with torch.no_grad():
        for label, path in _methods():
            policy = load_policy(path)
            for inst in instances:
                name = getattr(inst, "name", f"synth_n{size}")
                attempts, feasible, c_t, c_exec, c_req = _construct(
                    policy, inst, samples
                )
                if not feasible:
                    for mode in modes:
                        rows.append(
                            {
                                "instance": name,
                                "layout": name[:3],
                                "method": label,
                                "mode": mode,
                                "size": size,
                                "synthetic": int(synthetic),
                                "feasible": 0,
                                "objective_h": "",
                                "bks_gap_lit_pct": "",
                                "best_construct_h": "",
                                "mean_construct_h": "",
                                "runtime_s": round(c_t, 4),
                                "oracle_solver_executions": c_exec,
                                "oracle_requests": c_req,
                                "n_routes": "",
                                "n_charges": "",
                                "interior_insertions": "",
                            }
                        )
                    continue
                best = min(feasible, key=lambda a: a["objective"])
                construct_objs = [a["objective"] for a in feasible]
                start_routes = [tuple(r) for r in best["routes"]]
                start_plans = list(best["plans"])
                mode_rows = _run_modes(
                    policy, inst, start_routes, start_plans, modes
                )
                for mode, mr in zip(modes, mode_rows):
                    obj = mr["objective_h"]
                    lit = (
                        literature_compatible_objective_hours(inst, obj)
                        if mr["feasible"]
                        else float("inf")
                    )
                    gap = ""
                    if mr["feasible"] and name in bks:
                        gap = round(100.0 * (lit - bks[name]) / bks[name], 4)
                    n_charges = ""
                    if mr["feasible"]:
                        n_charges = sum(
                            getattr(p, "n_charges", 0)
                            if hasattr(p, "n_charges")
                            else sum(
                                1
                                for _, amt in (p.stops or ())
                                if amt is not None and amt > 1e-9
                            )
                            for p in mr["plans"]
                        )
                    diag = mr.get("diag") or {}
                    rows.append(
                        {
                            "instance": name,
                            "layout": name[:3],
                            "method": label,
                            "mode": mode,
                            "size": size,
                            "synthetic": int(synthetic),
                            "feasible": int(mr["feasible"]),
                            "objective_h": obj if mr["feasible"] else "",
                            "bks_gap_lit_pct": gap,
                            "best_construct_h": min(construct_objs),
                            "mean_construct_h": _mean(construct_objs),
                            "runtime_s": round(c_t + mr["runtime_s"], 4),
                            "oracle_solver_executions": c_exec
                            + mr["oracle_solver_executions"],
                            "oracle_requests": c_req + mr["oracle_requests"],
                            "n_routes": len(mr["routes"]) if mr["feasible"] else "",
                            "n_charges": n_charges,
                            "interior_insertions": diag.get(
                                "interior_position_insertions", ""
                            ),
                            "existing_route_insertions": diag.get(
                                "existing_route_insertions", ""
                            ),
                            "successful_repairs": diag.get("successful_repairs", ""),
                        }
                    )
                    print(
                        f"{'synth' if synthetic else 'off'} n={size} {mode} {label} "
                        f"{name} obj={obj if mr['feasible'] else 'inf'} gap={gap} "
                        f"exec={rows[-1]['oracle_solver_executions']}",
                        flush=True,
                    )
    out_tag = tag or ("synth" if synthetic else "official")
    _write_csv(RES / f"{PREFIX}_{out_tag}_n{size}.csv", rows)
    return rows


def _summarize(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by = defaultdict(
        lambda: {
            "obj": [],
            "gap": [],
            "rt": [],
            "ex": [],
            "n": 0,
            "f": 0,
            "interior": [],
        }
    )
    for r in rows:
        key = f"{r['method']}|{r['mode']}"
        by[key]["n"] += 1
        if r["feasible"] == "1":
            by[key]["f"] += 1
            by[key]["obj"].append(float(r["objective_h"]))
            if r["bks_gap_lit_pct"] != "":
                by[key]["gap"].append(float(r["bks_gap_lit_pct"]))
            by[key]["rt"].append(float(r["runtime_s"]))
            by[key]["ex"].append(float(r["oracle_solver_executions"]))
            if r.get("interior_insertions", "") not in ("", None):
                by[key]["interior"].append(float(r["interior_insertions"]))
    return {
        k: {
            "feas_pct": 100 * d["f"] / max(1, d["n"]),
            "mean_full_h": _mean(d["obj"]),
            "mean_lit_gap_pct": _mean(d["gap"]),
            "median_lit_gap_pct": _median(d["gap"]),
            "worst_lit_gap_pct": max(d["gap"]) if d["gap"] else float("nan"),
            "mean_runtime_s": _mean(d["rt"]),
            "mean_oracle_exec": _mean(d["ex"]),
            "mean_interior_insertions": _mean(d["interior"]),
        }
        for k, d in sorted(by.items())
    }


def stage_report() -> None:
    lines = [
        "# Insertion-Aware Neural Repair Report",
        "",
        "## Root cause",
        "",
        "Open-route neural repair still only **appended** customers via",
        "`oracle.solve((*prefix, customer))`. Classical ruin evaluates all",
        "insertion positions. Position-aware repair keeps the hierarchical",
        "policy for route/customer/new-route choice and uses",
        "`best_feasible_insertion()` (frvcpy at every position) for placement.",
        "",
        "## Code changes",
        "",
        "- `train.best_feasible_insertion`",
        "- `decode(..., repair_insertion=True|False, oracle=...)`",
        "- `neural_lns` / `hybrid_lns` / `repair_reinforce_step` use insertion mode",
        "- `FixedRouteOracle`: `requests`, `cache_hits`, `solver_executions`",
        "- Matched eval in `results/run_insertion_repair.py` (new CSVs only)",
        "",
        "## Regression tests",
        "",
        "```bash",
        "python -m unittest tests.test_insertion_repair tests.test_repair_and_validation -v",
        "```",
        "",
    ]
    for tag, size in (
        ("synth", 20),
        ("official", 10),
        ("official", 20),
        ("official", 40),
    ):
        path = RES / f"{PREFIX}_{tag}_n{size}.csv"
        if not path.exists():
            continue
        summary = _summarize(path)
        lines.append(f"## {tag} n={size} matched search")
        lines.append("")
        lines.append(
            "| Method | Mode | Feas% | Mean lit gap% | Median | Worst | Mean exec | Mean runtime |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for key, d in summary.items():
            method, mode = key.split("|", 1)
            lines.append(
                f"| {method} | {mode} | {d['feas_pct']:.1f} | "
                f"{d['mean_lit_gap_pct']:.3f} | {d['median_lit_gap_pct']:.3f} | "
                f"{d['worst_lit_gap_pct']:.3f} | {d['mean_oracle_exec']:.0f} | "
                f"{d['mean_runtime_s']:.2f} |"
            )
        lines.append("")
        # Paired neural_insert - classical
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        paired = defaultdict(dict)
        for r in rows:
            if r["feasible"] != "1" or r["bks_gap_lit_pct"] == "":
                continue
            paired[(r["method"], r["instance"])][r["mode"]] = float(
                r["bks_gap_lit_pct"]
            )
        lines.append(f"### Paired gap deltas ({tag} n={size}; + ⇒ classical better)")
        lines.append("")
        deltas = defaultdict(list)
        for (method, _), modes in paired.items():
            if "classical_ruin" in modes and "neural_insert" in modes:
                deltas[(method, "insert-classical")].append(
                    modes["neural_insert"] - modes["classical_ruin"]
                )
            if "classical_ruin" in modes and "neural_append" in modes:
                deltas[(method, "append-classical")].append(
                    modes["neural_append"] - modes["classical_ruin"]
                )
            if "neural_append" in modes and "neural_insert" in modes:
                deltas[(method, "insert-append")].append(
                    modes["neural_insert"] - modes["neural_append"]
                )
            if "classical_ruin" in modes and "hybrid_insert" in modes:
                deltas[(method, "hybrid-classical")].append(
                    modes["hybrid_insert"] - modes["classical_ruin"]
                )
        for (method, name), vals in sorted(deltas.items()):
            lines.append(
                f"- {method} {name}: mean {_mean(vals):.4f} pp (n={len(vals)})"
            )
        lines.append("")

    # Best configs
    best10 = (float("inf"), "")
    best20 = (float("inf"), "")
    for size, holder in ((10, "best10"), (20, "best20")):
        path = RES / f"{PREFIX}_official_n{size}.csv"
        if not path.exists():
            continue
        for key, d in _summarize(path).items():
            gap = d["mean_lit_gap_pct"]
            if math.isfinite(gap):
                if size == 10 and gap < best10[0]:
                    best10 = (gap, key)
                if size == 20 and gap < best20[0]:
                    best20 = (gap, key)
    lines.extend(
        [
            "## Conclusions",
            "",
            f"- Best official n=10: **{best10[1]}** (mean lit gap {best10[0]:.3f}%)",
            f"- Best official n=20: **{best20[1]}** (mean lit gap {best20[0]:.3f}%)",
            "- Architecture claims require matched H2 vs H4 settings.",
            "- Search gains labeled as search, not DeepSets.",
            "- Classical ruin remains default unless neural_insert/hybrid clearly wins",
            "  under matched oracle executions and wall-clock.",
            "",
            "## Commands",
            "",
            "```bash",
            "python results/run_insertion_repair.py --stage all",
            "```",
            "",
        ]
    )
    out = RES / "INSERTION_REPAIR_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "tests",
            "synth20",
            "official10",
            "official20",
            "official40",
            "report",
            "all",
        ),
        default="all",
    )
    args = parser.parse_args()
    t0 = time.perf_counter()
    if args.stage in ("tests", "all"):
        print("===== STAGE tests =====", flush=True)
        stage_tests()
    if args.stage in ("synth20", "all"):
        print("===== STAGE synth20 =====", flush=True)
        stage_eval(20, synthetic=True, samples=4, limit=8, tag="synth")
    if args.stage in ("official10", "all"):
        print("===== STAGE official10 =====", flush=True)
        stage_eval(10, samples=6)
    if args.stage in ("official20", "all"):
        print("===== STAGE official20 =====", flush=True)
        stage_eval(20, samples=4)
    if args.stage in ("official40", "all"):
        print("===== STAGE official40 =====", flush=True)
        stage_eval(40, samples=3, limit=20)
    if args.stage in ("report", "all"):
        print("===== STAGE report =====", flush=True)
        stage_report()
    print(f"===== done in {time.perf_counter() - t0:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()
