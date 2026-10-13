"""Nonlinear vs linear charging sensitivity (fixed-route + end-to-end).

Does not modify XML or overwrite Cycle-3 checkpoints.
All physically reported feasibility uses TRUE Montoya curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import (  # noqa: E402
    FixedRouteOracle,
    check_route,
    check_solution,
    decompose_solution,
)
from charging_surrogates import (  # noqa: E402
    document_surrogate,
    instance_with_charging_model,
)
from data import instance_files, parse_instance  # noqa: E402
from evaluate import _refine_best_once  # noqa: E402
from train import decode, load_policy  # noqa: E402

OUT = ROOT / "results" / "raw" / "nonlinear_charging"
CKPT = ROOT / "checkpoints" / "H2_n10_seed42_c3.pt"
SEED = 42
MODELS = ("TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE")


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


def _instances(size: int, limit: int | None = None):
    out = []
    for path in instance_files(ROOT / "dataset"):
        inst = parse_instance(path)
        if len(inst.customers) == size:
            out.append(inst)
        if limit is not None and len(out) >= limit:
            break
    return out


def _customer_routes_from_plans(true_inst, routes, plans):
    """Prefer validated customer sequences from plans; fall back to routes."""
    seqs = []
    for route, plan in zip(routes, plans):
        if plan is not None and plan.feasible and plan.stops is not None:
            customers = tuple(
                n for n, _ in plan.stops if true_inst.nodes[n].kind == 1
            )
            seqs.append(customers)
        else:
            seqs.append(tuple(route))
    return seqs


def obtain_reference_sequences(policy, inst):
    """Matched construction: H2 greedy + basic LS under TRUE-NL."""
    with torch.no_grad():
        out = decode(policy, inst, sample=False, seed=SEED, cost_lambda=0.0)
    if not out["feasible"]:
        # try few samples
        attempts = [
            decode(policy, inst, sample=True, seed=SEED * 1000 + i, cost_lambda=0.0)
            for i in range(5)
        ]
        feas = [a for a in attempts if a["feasible"]]
        if not feas:
            return None
        out = min(feas, key=lambda a: a["objective"])
    routes, plans, _, _, _ = _refine_best_once(
        inst, list(out["routes"]), list(out["plans"]), 3, 4000, "basic"
    )
    valid, reason = check_solution(inst, plans)
    if not valid:
        return None
    return {
        "routes": [tuple(r) for r in routes],
        "plans": plans,
        "sequences": _customer_routes_from_plans(inst, routes, plans),
        "true_obj": sum(p.duration_hours for p in plans),
    }


def evaluate_stops_true(true_inst, sequences, stops_list):
    """Physical evaluation of planned stops under TRUE-NL curves."""
    if any(s is None for s in stops_list):
        return {
            "feasible": False,
            "reason": "missing_stops",
            "full_h": "",
            "travel_h": "",
            "service_h": "",
            "charging_h": "",
            "lit_h": "",
            "energy_kwh": "",
            "n_charges": "",
            "n_routes": len(sequences),
            "duration_violation": "",
            "soc_fail": "",
        }
    # Build RoutePlan-like objects for check_solution
    from charging import RoutePlan

    plans = []
    duration_violation = 0
    soc_fail = 0
    for seq, stops in zip(sequences, stops_list):
        check = check_route(true_inst, stops, seq)
        plans.append(
            RoutePlan(check.feasible, check.duration_hours, stops, check.reason)
        )
        if not check.feasible:
            if "duration" in check.reason.lower() or "tmax" in check.reason.lower() or "max" in check.reason.lower():
                duration_violation = 1
            if "soc" in check.reason.lower() or "energy" in check.reason.lower() or "battery" in check.reason.lower():
                soc_fail = 1
    valid, reason = check_solution(true_inst, plans)
    if not valid:
        return {
            "feasible": False,
            "reason": reason,
            "full_h": "",
            "travel_h": "",
            "service_h": "",
            "charging_h": "",
            "lit_h": "",
            "energy_kwh": "",
            "n_charges": sum(
                check_route(true_inst, s).charging_stops for s in stops_list if s
            ),
            "n_routes": len(sequences),
            "duration_violation": duration_violation,
            "soc_fail": soc_fail,
        }
    decomp = decompose_solution(true_inst, plans)
    return {
        "feasible": True,
        "reason": "",
        "full_h": decomp["full_h"],
        "travel_h": decomp["travel_h"],
        "service_h": decomp["service_h"],
        "charging_h": decomp["charging_h"],
        "lit_h": decomp["lit_h"],
        "energy_kwh": decomp["energy_charged_wh"] / 1000.0,
        "n_charges": decomp["n_charges"],
        "n_routes": decomp["n_routes"],
        "duration_violation": 0,
        "soc_fail": 0,
        "station_types": "|".join(decomp["station_types"]),
        "identity_ok": int(decomp["identity_ok"]),
    }


def stage_document_curves():
    inst = _instances(10, limit=1)[0]
    rows = []
    for model in MODELS:
        rows.extend(document_surrogate(inst, model))
    _write_csv(OUT / "surrogate_definitions.csv", rows)


def stage_fixed_route(sizes=(10, 20, 40), limit: int | None = None):
    policy = load_policy(CKPT)
    rows = []
    schedule_dir = OUT / "schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for size in sizes:
            instances = _instances(size, limit=limit)
            print(f"===== fixed-route size={size} n={len(instances)} =====", flush=True)
            for inst in instances:
                ref = obtain_reference_sequences(policy, inst)
                if ref is None:
                    print(f"skip infeasible construction {inst.name}", flush=True)
                    continue
                sequences = ref["sequences"]
                # TRUE-NL optimization on fixed sequences
                true_oracle = FixedRouteOracle(inst)
                true_plans = [true_oracle.solve(seq) for seq in sequences]
                true_stops = [p.stops if p.feasible else None for p in true_plans]
                true_eval = evaluate_stops_true(inst, sequences, true_stops)

                row_base = {
                    "experiment": "fixed_route",
                    "instance": inst.name,
                    "size": size,
                    "n_customers": len(inst.customers),
                    "seed": SEED,
                    "checkpoint": CKPT.name,
                    "n_sequences": len(sequences),
                }
                rows.append(
                    {
                        **row_base,
                        "charging_model": "TRUE-NL",
                        "method": "plan_TRUE-NL",
                        "feasible_under_planner": int(
                            all(p.feasible for p in true_plans)
                        ),
                        "feasible_under_true_physics": int(true_eval["feasible"]),
                        "failure_reason": true_eval["reason"],
                        "duration_violation": true_eval["duration_violation"],
                        "soc_fail": true_eval["soc_fail"],
                        "nominal_planner_charge_h": (
                            sum(
                                check_route(inst, p.stops).charging_hours
                                for p in true_plans
                                if p.feasible
                            )
                            if all(p.feasible for p in true_plans)
                            else ""
                        ),
                        "true_charging_h": true_eval["charging_h"],
                        "travel_h": true_eval["travel_h"],
                        "service_h": true_eval["service_h"],
                        "full_h_true": true_eval["full_h"],
                        "lit_h_true": true_eval["lit_h"],
                        "energy_kwh": true_eval["energy_kwh"],
                        "n_charges": true_eval["n_charges"],
                        "n_routes": true_eval["n_routes"],
                        "oracle_solver_executions": true_oracle.solver_executions,
                        "runtime_s": "",
                    }
                )
                if true_eval["feasible"]:
                    (schedule_dir / f"{inst.name}_TRUE-NL.json").write_text(
                        json.dumps(
                            {
                                "instance": inst.name,
                                "model": "TRUE-NL",
                                "sequences": [list(s) for s in sequences],
                                "stops": [
                                    [
                                        [int(n), None if a is None else float(a)]
                                        for n, a in p.stops
                                    ]
                                    for p in true_plans
                                ],
                                "full_h": true_eval["full_h"],
                            }
                        ),
                        encoding="utf-8",
                    )

                for model in ("LINEAR-SECANT", "LINEAR-EARLY-RATE"):
                    t0 = time.perf_counter()
                    lin_inst = instance_with_charging_model(inst, model)
                    lin_oracle = FixedRouteOracle(lin_inst)
                    lin_plans = [lin_oracle.solve(seq) for seq in sequences]
                    planner_feas = all(p.feasible for p in lin_plans)
                    lin_stops = [p.stops if p.feasible else None for p in lin_plans]
                    # Nominal charging under surrogate
                    nominal_charge = ""
                    if planner_feas:
                        nominal_charge = sum(
                            check_route(lin_inst, p.stops).charging_hours
                            for p in lin_plans
                        )
                    true_eval_lin = evaluate_stops_true(inst, sequences, lin_stops)
                    dt = time.perf_counter() - t0
                    rows.append(
                        {
                            **row_base,
                            "charging_model": model,
                            "method": f"plan_{model}",
                            "feasible_under_planner": int(planner_feas),
                            "feasible_under_true_physics": int(
                                true_eval_lin["feasible"]
                            ),
                            "failure_reason": true_eval_lin["reason"],
                            "duration_violation": true_eval_lin["duration_violation"],
                            "soc_fail": true_eval_lin["soc_fail"],
                            "nominal_planner_charge_h": nominal_charge,
                            "true_charging_h": true_eval_lin["charging_h"],
                            "travel_h": true_eval_lin["travel_h"],
                            "service_h": true_eval_lin["service_h"],
                            "full_h_true": true_eval_lin["full_h"],
                            "lit_h_true": true_eval_lin["lit_h"],
                            "energy_kwh": true_eval_lin["energy_kwh"],
                            "n_charges": true_eval_lin["n_charges"],
                            "n_routes": true_eval_lin["n_routes"],
                            "oracle_solver_executions": lin_oracle.solver_executions,
                            "runtime_s": round(dt, 4),
                        }
                    )
                    # Separate repaired method: re-optimize sequences with TRUE-NL
                    t1 = time.perf_counter()
                    repair_oracle = FixedRouteOracle(inst)
                    repair_plans = [repair_oracle.solve(seq) for seq in sequences]
                    repair_stops = [
                        p.stops if p.feasible else None for p in repair_plans
                    ]
                    repair_eval = evaluate_stops_true(inst, sequences, repair_stops)
                    rows.append(
                        {
                            **row_base,
                            "charging_model": model,
                            "method": f"repair_TRUE-NL_after_{model}",
                            "feasible_under_planner": int(
                                all(p.feasible for p in repair_plans)
                            ),
                            "feasible_under_true_physics": int(repair_eval["feasible"]),
                            "failure_reason": repair_eval["reason"],
                            "duration_violation": repair_eval["duration_violation"],
                            "soc_fail": repair_eval["soc_fail"],
                            "nominal_planner_charge_h": "",
                            "true_charging_h": repair_eval["charging_h"],
                            "travel_h": repair_eval["travel_h"],
                            "service_h": repair_eval["service_h"],
                            "full_h_true": repair_eval["full_h"],
                            "lit_h_true": repair_eval["lit_h"],
                            "energy_kwh": repair_eval["energy_kwh"],
                            "n_charges": repair_eval["n_charges"],
                            "n_routes": repair_eval["n_routes"],
                            "oracle_solver_executions": repair_oracle.solver_executions,
                            "runtime_s": round(time.perf_counter() - t1, 4),
                        }
                    )
                    if not true_eval_lin["feasible"] and lin_stops[0] is not None:
                        (schedule_dir / f"{inst.name}_{model}_infeasible_under_true.json").write_text(
                            json.dumps(
                                {
                                    "instance": inst.name,
                                    "model": model,
                                    "sequences": [list(s) for s in sequences],
                                    "stops": [
                                        [
                                            [int(n), None if a is None else float(a)]
                                            for n, a in (stops or [])
                                        ]
                                        for stops in lin_stops
                                    ],
                                    "true_reason": true_eval_lin["reason"],
                                }
                            ),
                            encoding="utf-8",
                        )
                    print(
                        f"fixed {inst.name} {model}: planner={planner_feas} "
                        f"true_phys={true_eval_lin['feasible']} reason={true_eval_lin['reason']}",
                        flush=True,
                    )
    _write_csv(OUT / "fixed_route_charging_comparison.csv", rows)
    return rows


def stage_end_to_end(sizes=(10, 20, 40), limit: int | None = None):
    """Same neural weights; planning oracle uses different charging models."""
    policy = load_policy(CKPT)
    rows = []
    with torch.no_grad():
        for size in sizes:
            instances = _instances(size, limit=limit)
            print(f"===== e2e size={size} n={len(instances)} =====", flush=True)
            for inst in instances:
                for model in MODELS:
                    t0 = time.perf_counter()
                    plan_inst = instance_with_charging_model(inst, model)
                    # Construction under planning physics
                    out = decode(
                        policy, plan_inst, sample=False, seed=SEED, cost_lambda=0.0
                    )
                    if not out["feasible"]:
                        attempts = [
                            decode(
                                policy,
                                plan_inst,
                                sample=True,
                                seed=SEED * 1000 + i,
                                cost_lambda=0.0,
                            )
                            for i in range(5)
                        ]
                        feas = [a for a in attempts if a["feasible"]]
                        out = min(feas, key=lambda a: a["objective"]) if feas else out
                    planner_feas = bool(out["feasible"])
                    sequences = (
                        [tuple(r) for r in out["routes"]] if planner_feas else []
                    )
                    stops_list = (
                        [p.stops for p in out["plans"]] if planner_feas else []
                    )
                    # Re-evaluate planned stops under TRUE physics
                    if planner_feas:
                        true_eval = evaluate_stops_true(inst, sequences, stops_list)
                    else:
                        true_eval = {
                            "feasible": False,
                            "reason": "planner_infeasible",
                            "full_h": "",
                            "travel_h": "",
                            "service_h": "",
                            "charging_h": "",
                            "lit_h": "",
                            "energy_kwh": "",
                            "n_charges": "",
                            "n_routes": "",
                            "duration_violation": "",
                            "soc_fail": "",
                        }
                    # Route fingerprint for decision change analysis
                    fingerprint = (
                        "|".join("-".join(str(c) for c in seq) for seq in sequences)
                        if sequences
                        else ""
                    )
                    rows.append(
                        {
                            "experiment": "end_to_end",
                            "instance": inst.name,
                            "size": size,
                            "charging_model": model,
                            "checkpoint": CKPT.name,
                            "seed": SEED,
                            "feasible_under_planner": int(planner_feas),
                            "feasible_under_true_physics": int(true_eval["feasible"]),
                            "failure_reason": true_eval["reason"],
                            "duration_violation": true_eval.get("duration_violation", ""),
                            "full_h_true": true_eval["full_h"],
                            "travel_h": true_eval["travel_h"],
                            "service_h": true_eval["service_h"],
                            "true_charging_h": true_eval["charging_h"],
                            "lit_h_true": true_eval["lit_h"],
                            "energy_kwh": true_eval["energy_kwh"],
                            "n_charges": true_eval["n_charges"],
                            "n_routes": true_eval["n_routes"],
                            "route_fingerprint": fingerprint,
                            "runtime_s": round(time.perf_counter() - t0, 4),
                        }
                    )
                    print(
                        f"e2e {inst.name} {model}: planner={planner_feas} "
                        f"true={true_eval['feasible']}",
                        flush=True,
                    )
    _write_csv(OUT / "end_to_end_charging_sensitivity.csv", rows)
    # Decision change summary vs TRUE-NL
    by_inst = {}
    for r in rows:
        by_inst.setdefault((r["size"], r["instance"]), {})[r["charging_model"]] = r
    changes = []
    for (size, name), d in sorted(by_inst.items()):
        if "TRUE-NL" not in d:
            continue
        base_fp = d["TRUE-NL"]["route_fingerprint"]
        for model in ("LINEAR-SECANT", "LINEAR-EARLY-RATE"):
            if model not in d:
                continue
            changes.append(
                {
                    "size": size,
                    "instance": name,
                    "charging_model": model,
                    "route_changed": int(
                        d[model]["route_fingerprint"] != base_fp
                        and d[model]["route_fingerprint"] != ""
                        and base_fp != ""
                    ),
                    "true_nl_feasible": d["TRUE-NL"]["feasible_under_true_physics"],
                    "linear_true_feasible": d[model]["feasible_under_true_physics"],
                    "true_nl_full_h": d["TRUE-NL"]["full_h_true"],
                    "linear_full_h_true": d[model]["full_h_true"],
                }
            )
    _write_csv(OUT / "end_to_end_decision_changes.csv", changes)
    return rows


def stage_summarize():
    fixed = list(
        csv.DictReader((OUT / "fixed_route_charging_comparison.csv").open(encoding="utf-8"))
    ) if (OUT / "fixed_route_charging_comparison.csv").exists() else []
    e2e = list(
        csv.DictReader((OUT / "end_to_end_charging_sensitivity.csv").open(encoding="utf-8"))
    ) if (OUT / "end_to_end_charging_sensitivity.csv").exists() else []

    def agg(rows, experiment):
        out = []
        keys = set((r["size"], r["charging_model"], r["method"]) for r in rows) if rows and "method" in rows[0] else set((r["size"], r["charging_model"]) for r in rows)
        if rows and "method" in rows[0]:
            groups = {}
            for r in rows:
                groups.setdefault((r["size"], r["charging_model"], r["method"]), []).append(r)
        else:
            groups = {}
            for r in rows:
                groups.setdefault((r["size"], r["charging_model"], "e2e"), []).append(r)
        for key, group in sorted(groups.items()):
            size, model, method = key
            n = len(group)
            feas_true = sum(int(r["feasible_under_true_physics"]) for r in group)
            feas_plan = sum(int(r["feasible_under_planner"]) for r in group)
            dur_v = sum(int(r["duration_violation"] or 0) for r in group if r.get("duration_violation") != "")
            feas_rows = [r for r in group if r["feasible_under_true_physics"] == "1"]
            def mean_key(k):
                vals = [float(r[k]) for r in feas_rows if r.get(k) not in ("", None)]
                return sum(vals) / len(vals) if vals else ""
            out.append(
                {
                    "experiment": experiment,
                    "size": size,
                    "charging_model": model,
                    "method": method,
                    "n_instances": n,
                    "feasible_under_planner": feas_plan,
                    "feasible_under_true_physics": feas_true,
                    "duration_violations": dur_v,
                    "mean_true_charging_h_feasible_only": mean_key("true_charging_h"),
                    "mean_full_h_true_feasible_only": mean_key("full_h_true"),
                    "mean_energy_kwh_feasible_only": mean_key("energy_kwh"),
                    "mean_n_charges_feasible_only": mean_key("n_charges"),
                    "note": "feasible-only means exclude infeasible; do not rank by them when feas rates differ",
                }
            )
        return out

    summary = agg(fixed, "fixed_route") + agg(e2e, "end_to_end")
    _write_csv(OUT / "charging_study_summary.csv", summary)

    # Markdown study report
    lines = [
        "# Nonlinear Charging Study",
        "",
        "Priority experiment: consequences of replacing Montoya nonlinear charging",
        "with explicit linear surrogates, evaluated under **true** physics.",
        "",
        "## Surrogates",
        "",
        "- **TRUE-NL**: station-specific piecewise-linear Montoya curves + frvcpy.",
        "- **LINEAR-SECANT**: constant rate on the empty→full chord for each technology.",
        "- **LINEAR-EARLY-RATE**: constant rate from the first true breakpoint segment, extrapolated to Q (optimistic for concave curves).",
        "",
        "Original XML files are never modified. Surrogates replace `Instance.curves` only for planning.",
        "",
        "## Fixed-route protocol",
        "",
        "1. Obtain customer sequences with H2 Cycle-3 + basic LS under TRUE-NL.",
        "2. Optimize charging independently under each model for those fixed sequences.",
        "3. Re-evaluate every planned stop list with the true nonlinear checker.",
        "4. Separately report TRUE-NL re-optimization of the same sequences after a linear plan (`repair_TRUE-NL_after_*`).",
        "",
        "## Summary (raw: `charging_study_summary.csv`)",
        "",
        "| Exp | n | Model | Method | N | Feas planner | Feas true physics | Dur violations |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['experiment']} | {r['size']} | {r['charging_model']} | {r['method']} | "
            f"{r['n_instances']} | {r['feasible_under_planner']} | "
            f"{r['feasible_under_true_physics']} | {r['duration_violations']} |"
        )
    lines.extend(
        [
            "",
            "## Raw outputs",
            "",
            "- `fixed_route_charging_comparison.csv`",
            "- `end_to_end_charging_sensitivity.csv`",
            "- `end_to_end_decision_changes.csv`",
            "- `surrogate_definitions.csv`",
            "- `schedules/`",
            "",
            "## Commands",
            "",
            "```bash",
            "python scripts/experiments/run_nonlinear_charging_study.py --stage all",
            "```",
            "",
        ]
    )
    report = ROOT / "results" / "NONLINEAR_CHARGING_STUDY.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("doc", "fixed", "e2e", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--sizes", type=str, default="10,20,40")
    parser.add_argument("--limit", type=int, default=None, help="instances per size")
    args = parser.parse_args()
    sizes = tuple(int(x) for x in args.sizes.split(",") if x.strip())
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("doc", "all"):
        stage_document_curves()
    if args.stage in ("fixed", "all"):
        stage_fixed_route(sizes=sizes, limit=args.limit)
    if args.stage in ("e2e", "all"):
        stage_end_to_end(sizes=sizes, limit=args.limit)
    if args.stage in ("summarize", "all"):
        stage_summarize()


if __name__ == "__main__":
    main()
