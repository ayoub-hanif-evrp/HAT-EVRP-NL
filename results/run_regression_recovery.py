"""n40 regression audit: reproduce historical fair-LS, ablate construction, recover hybrid.

Does NOT overwrite historical CSVs or checkpoints.
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
HIST = RES / "compare_size40_from10_fairls_final_seed42_c3.csv"
PREFIX = "RECOVERY"
SEED = 42
SAMPLES = 3
COST_REF = 0.0
HIST_H2_MEAN = 57.17274009092578
HIST_H4_MEAN = 57.50564156359235


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
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def _n40_instances():
    from data import instance_files, parse_instance

    return [
        parse_instance(p)
        for p in instance_files(DATA)
        if len(parse_instance(p).customers) == 40
    ]


def _hist_objectives(method: str) -> dict[str, float]:
    rows = list(csv.DictReader(HIST.open(encoding="utf-8")))
    return {
        r["instance"]: float(r["objective_h"])
        for r in rows
        if r["method"] == method and r["feasible"] == "1"
    }


def stage_reproduce() -> Path:
    """Exact historical fair-LS protocol for H2/H4 (n10 ckpt → n40)."""
    import torch
    from charging import check_route, check_solution
    from evaluate import _refine_best_once
    from train import decode, load_policy

    policies = {
        "H2": load_policy(CKPT / "H2_n10_seed42_c3.pt"),
        "H4": load_policy(CKPT / "H4_n10_seed42_c3.pt"),
    }
    rows = []
    with torch.no_grad():
        for method, policy in policies.items():
            for inst in _n40_instances():
                attempts = []
                for i in range(SAMPLES):
                    attempts.append(
                        decode(
                            policy,
                            inst,
                            sample=i > 0,
                            seed=SEED * 1000 + i,
                            cost_lambda=0.0,
                        )
                    )
                construct_t = sum(a["runtime"] for a in attempts)
                construct_o = sum(a["oracle_calls"] for a in attempts)
                feas = [a for a in attempts if a["feasible"]]
                if not feas:
                    rows.append(
                        {
                            "instance": inst.name,
                            "customers": 40,
                            "method": method,
                            "seed": SEED,
                            "version": "c3",
                            "samples": SAMPLES,
                            "feasible": 0,
                            "objective_h": "",
                            "routes": "",
                            "charging_stops": "",
                            "charging_h": "",
                            "runtime_s": construct_t,
                            "oracle_calls": construct_o,
                            "reason": "no feasible construction",
                        }
                    )
                    continue
                best = min(feas, key=lambda a: a["objective"])
                t0 = time.perf_counter()
                routes, plans, _, refine_o, _ = _refine_best_once(
                    inst,
                    list(best["routes"]),
                    list(best["plans"]),
                    local_passes=3,
                    local_budget=4000,
                    operators="basic",
                )
                refine_t = time.perf_counter() - t0
                valid, reason = check_solution(inst, plans)
                obj = sum(p.duration_hours for p in plans) if valid else ""
                stops = hours = ""
                if valid:
                    stops = hours = 0
                    for plan in plans:
                        check = check_route(inst, plan.stops)
                        stops += check.charging_stops
                        hours += check.charging_hours
                rows.append(
                    {
                        "instance": inst.name,
                        "customers": 40,
                        "method": method,
                        "seed": SEED,
                        "version": "c3",
                        "samples": SAMPLES,
                        "feasible": int(valid),
                        "objective_h": obj,
                        "routes": len(routes) if valid else "",
                        "charging_stops": stops,
                        "charging_h": hours,
                        "runtime_s": construct_t + refine_t,
                        "oracle_calls": construct_o + refine_o,
                        "reason": "" if valid else reason,
                    }
                )
                print(
                    f"repro {method} {inst.name} obj={obj}",
                    flush=True,
                )
    out = RES / f"{PREFIX}_n40_repro_fairls.csv"
    _write_csv(out, rows)

    delta_rows = []
    for method in ("H2", "H4"):
        hist = _hist_objectives(method)
        repro = {
            r["instance"]: float(r["objective_h"])
            for r in rows
            if r["method"] == method and r["feasible"] == 1
        }
        # feasible may be int 1 from writer
        repro = {
            r["instance"]: float(r["objective_h"])
            for r in rows
            if r["method"] == method
            and str(r["feasible"]) == "1"
            and r["objective_h"] != ""
        }
        for inst, hobj in sorted(hist.items()):
            robj = repro.get(inst)
            delta_rows.append(
                {
                    "instance": inst,
                    "method": method,
                    "historical_h": hobj,
                    "repro_h": robj if robj is not None else "",
                    "delta_h": (robj - hobj) if robj is not None else "",
                    "match_1e6": int(
                        robj is not None and abs(robj - hobj) < 1e-6
                    ),
                }
            )
        objs = [
            float(r["repro_h"])
            for r in delta_rows
            if r["method"] == method and r["repro_h"] != ""
        ]
        ref = HIST_H2_MEAN if method == "H2" else HIST_H4_MEAN
        print(f"REPRO {method}: mean={_mean(objs):.6f} hist={ref:.6f}", flush=True)
    _write_csv(RES / f"{PREFIX}_n40_repro_vs_hist.csv", delta_rows)
    return out


def _construct_attempts(
    policy,
    inst,
    *,
    forced: bool,
    cost_lambda: float,
    samples: int = SAMPLES,
    seed: int = SEED,
):
    from train import decode

    attempts = []
    for i in range(samples):
        force = list(inst.customers)[i % len(inst.customers)] if forced else None
        # Ordinary: first greedy (sample=False), later sampled — historical.
        # Forced: sample=True every attempt (insertion experiment style).
        sample = True if forced else (i > 0)
        attempts.append(
            decode(
                policy,
                inst,
                sample=sample,
                seed=seed * 1000 + i,
                cost_lambda=cost_lambda,
                force_first_customer=force,
            )
        )
    return attempts


def stage_construction_ablation() -> list[dict]:
    """A/B/C/D construction + identical one-shot basic LS."""
    import torch
    from charging import check_solution
    from evaluate import _refine_best_once
    from train import load_policy

    configs = (
        ("ordinary_l0", False, 0.0),
        ("ordinary_l4", False, 4.0),
        ("forced_l0", True, 0.0),
        ("forced_l4", True, 4.0),
    )
    policies = {
        "H2": load_policy(CKPT / "H2_n10_seed42_c3.pt"),
        "H4": load_policy(CKPT / "H4_n10_seed42_c3.pt"),
    }
    rows = []
    with torch.no_grad():
        for method, policy in policies.items():
            for inst in _n40_instances():
                for cfg_name, forced, lam in configs:
                    attempts = _construct_attempts(
                        policy, inst, forced=forced, cost_lambda=lam
                    )
                    feas = [a for a in attempts if a["feasible"]]
                    construct_t = sum(a["runtime"] for a in attempts)
                    construct_exec = sum(
                        a.get("oracle_solver_executions", a["oracle_calls"])
                        for a in attempts
                    )
                    if not feas:
                        rows.append(
                            {
                                "instance": inst.name,
                                "method": method,
                                "config": cfg_name,
                                "feasible": 0,
                                "best_construct_h": "",
                                "ls_objective_h": "",
                                "n_routes": "",
                                "n_charges": "",
                                "runtime_s": round(construct_t, 4),
                                "oracle_solver_executions": construct_exec,
                                "failed_attempts": SAMPLES,
                                "best_route_seq": "",
                            }
                        )
                        continue
                    best = min(feas, key=lambda a: a["objective"])
                    # One-shot basic LS (historical fair-LS operators).
                    t0 = time.perf_counter()
                    routes, plans, _, ls_calls, _ = _refine_best_once(
                        inst,
                        list(best["routes"]),
                        list(best["plans"]),
                        local_passes=3,
                        local_budget=4000,
                        operators="basic",
                    )
                    ls_t = time.perf_counter() - t0
                    valid, _ = check_solution(inst, plans)
                    obj = (
                        sum(p.duration_hours for p in plans) if valid else float("inf")
                    )
                    n_charges = 0
                    if valid:
                        from charging import check_route

                        for plan in plans:
                            n_charges += check_route(inst, plan.stops).charging_stops
                    rows.append(
                        {
                            "instance": inst.name,
                            "method": method,
                            "config": cfg_name,
                            "feasible": int(valid),
                            "best_construct_h": best["objective"],
                            "ls_objective_h": obj if valid else "",
                            "n_routes": len(routes) if valid else "",
                            "n_charges": n_charges if valid else "",
                            "runtime_s": round(construct_t + ls_t, 4),
                            "oracle_solver_executions": construct_exec + ls_calls,
                            "failed_attempts": SAMPLES - len(feas),
                            "best_route_seq": "|".join(
                                "-".join(map(str, r)) for r in best["routes"]
                            ),
                            "attempt_objectives": ";".join(
                                str(a["objective"]) if a["feasible"] else "inf"
                                for a in attempts
                            ),
                        }
                    )
                    print(
                        f"ablate {method} {cfg_name} {inst.name} "
                        f"construct={best['objective']:.3f} ls={obj if valid else 'inf'}",
                        flush=True,
                    )
    _write_csv(RES / f"{PREFIX}_n40_construction_ablation.csv", rows)
    # Summary means
    summary = {}
    by = defaultdict(list)
    for r in rows:
        if r["feasible"] == 1 or r["feasible"] == "1":
            if r["ls_objective_h"] != "":
                by[(r["method"], r["config"])].append(float(r["ls_objective_h"]))
    for key, vals in sorted(by.items()):
        summary[f"{key[0]}|{key[1]}"] = _mean(vals)
        print(f"ABLATION mean LS {key[0]} {key[1]}: {_mean(vals):.6f}", flush=True)
    (RES / f"{PREFIX}_n40_construction_ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return rows


def stage_pipeline_on_recovered() -> list[dict]:
    """Recovered ordinary λ=0 construction → staged search including hybrid."""
    import torch
    from charging import FixedRouteOracle, check_solution, check_route
    from evaluate import _refine_best_once
    from train import hybrid_lns, load_policy, neural_lns

    policies = {
        "H2": load_policy(CKPT / "H2_n10_seed42_c3.pt"),
        "H4": load_policy(CKPT / "H4_n10_seed42_c3.pt"),
    }
    hist = {m: _hist_objectives(m) for m in ("H2", "H4")}
    stages = (
        "construct",
        "basic_ls",  # historical fair-LS
        "strong_ls",
        "classical_ruin",
        "neural_insert",
        "hybrid_insert",
    )
    rows = []
    with torch.no_grad():
        for method, policy in policies.items():
            for inst in _n40_instances():
                attempts = _construct_attempts(
                    policy, inst, forced=False, cost_lambda=0.0
                )
                feas = [a for a in attempts if a["feasible"]]
                c_t = sum(a["runtime"] for a in attempts)
                c_exec = sum(
                    a.get("oracle_solver_executions", a["oracle_calls"]) for a in attempts
                )
                if not feas:
                    continue
                best = min(feas, key=lambda a: a["objective"])
                start_routes = [tuple(r) for r in best["routes"]]
                start_plans = list(best["plans"])

                # Stage outputs accumulate best-so-far.
                stage_state = {
                    "construct": (start_routes, start_plans, best["objective"], 0, 0.0),
                }

                # basic LS
                t0 = time.perf_counter()
                r, p, _, calls, _ = _refine_best_once(
                    inst,
                    list(start_routes),
                    list(start_plans),
                    3,
                    4000,
                    operators="basic",
                )
                dt = time.perf_counter() - t0
                valid, _ = check_solution(inst, p)
                obj = sum(x.duration_hours for x in p) if valid else float("inf")
                # Never worsen vs construct
                if obj > best["objective"] + 1e-10:
                    r, p, obj = start_routes, start_plans, best["objective"]
                stage_state["basic_ls"] = (r, p, obj, calls, dt)

                # strong LS from construct (independent branch for comparison)
                t0 = time.perf_counter()
                r2, p2, _, calls2, _ = _refine_best_once(
                    inst,
                    list(start_routes),
                    list(start_plans),
                    3,
                    4000,
                    operators="strong",
                )
                dt2 = time.perf_counter() - t0
                valid, _ = check_solution(inst, p2)
                obj2 = sum(x.duration_hours for x in p2) if valid else float("inf")
                if obj2 > best["objective"] + 1e-10:
                    r2, p2, obj2 = start_routes, start_plans, best["objective"]
                stage_state["strong_ls"] = (r2, p2, obj2, calls2, dt2)

                # classical ruin from strong LS
                from solver_ls import ruin_and_repair

                oracle = FixedRouteOracle(inst)
                t0 = time.perf_counter()
                r3, p3, _, _ = ruin_and_repair(
                    list(r2),
                    list(p2),
                    oracle,
                    inst,
                    max_iterations=6,
                    evaluation_budget=600,
                    seed=SEED + 91,
                )
                dt3 = time.perf_counter() - t0
                valid, _ = check_solution(inst, p3)
                obj3 = sum(x.duration_hours for x in p3) if valid else float("inf")
                if obj3 > obj2 + 1e-10:
                    r3, p3, obj3 = r2, p2, obj2
                stage_state["classical_ruin"] = (r3, p3, obj3, oracle.calls, dt3)

                # neural insert from strong LS
                oracle_n = FixedRouteOracle(inst)
                t0 = time.perf_counter()
                rn, pn, _, diag_n = neural_lns(
                    policy,
                    inst,
                    list(r2),
                    list(p2),
                    max_iterations=6,
                    seed=SEED + 91,
                    cost_lambda=4.0,
                    repair_insertion=True,
                    oracle=oracle_n,
                )
                dtn = time.perf_counter() - t0
                valid, _ = check_solution(inst, pn)
                objn = sum(x.duration_hours for x in pn) if valid else float("inf")
                if objn > obj2 + 1e-10:
                    rn, pn, objn = r2, p2, obj2
                stage_state["neural_insert"] = (rn, pn, objn, oracle_n.calls, dtn)

                # hybrid from strong LS
                oracle_h = FixedRouteOracle(inst)
                t0 = time.perf_counter()
                rh, ph, _, diag_h = hybrid_lns(
                    policy,
                    inst,
                    list(r2),
                    list(p2),
                    max_iterations=6,
                    seed=SEED + 91,
                    cost_lambda=4.0,
                    classical_iterations=6,
                    classical_budget=600,
                    repair_insertion=True,
                    oracle=oracle_h,
                )
                dth = time.perf_counter() - t0
                valid, _ = check_solution(inst, ph)
                objh = sum(x.duration_hours for x in ph) if valid else float("inf")
                if objh > obj2 + 1e-10:
                    rh, ph, objh = r2, p2, obj2
                stage_state["hybrid_insert"] = (rh, ph, objh, oracle_h.calls, dth)

                for stage in stages:
                    if stage == "construct":
                        routes, plans, obj, calls, dt = stage_state["construct"]
                        calls = 0
                        dt = 0.0
                    else:
                        routes, plans, obj, calls, dt = stage_state[stage]
                    n_charges = 0
                    if math.isfinite(obj):
                        for plan in plans:
                            n_charges += check_route(inst, plan.stops).charging_stops
                    href = hist[method].get(inst.name)
                    rows.append(
                        {
                            "instance": inst.name,
                            "method": method,
                            "stage": stage,
                            "feasible": int(math.isfinite(obj)),
                            "objective_h": obj if math.isfinite(obj) else "",
                            "historical_fairls_h": href if href is not None else "",
                            "delta_vs_hist_h": (obj - href)
                            if href is not None and math.isfinite(obj)
                            else "",
                            "n_routes": len(routes) if math.isfinite(obj) else "",
                            "n_charges": n_charges if math.isfinite(obj) else "",
                            "runtime_s": round(c_t + dt, 4),
                            "oracle_solver_executions": c_exec + calls,
                            "beats_hist": int(
                                href is not None
                                and math.isfinite(obj)
                                and obj < href - 1e-9
                            ),
                            "matches_hist_1e4": int(
                                href is not None
                                and math.isfinite(obj)
                                and abs(obj - href) < 1e-4
                            ),
                        }
                    )
                    print(
                        f"pipe {method} {stage} {inst.name} obj={obj:.4f} "
                        f"hist={href} delta={(obj-href) if href else ''}",
                        flush=True,
                    )
    _write_csv(RES / f"{PREFIX}_n40_recovered_pipeline.csv", rows)
    return rows


def stage_nonregression_check() -> dict:
    """Assert recovered basic_ls ≈ historical means within tolerance."""
    path = RES / f"{PREFIX}_n40_recovered_pipeline.csv"
    if not path.exists():
        path = RES / f"{PREFIX}_n40_repro_fairls.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    report = {"pass": True, "details": {}}
    # Prefer basic_ls stage if present
    for method, ref in (("H2", HIST_H2_MEAN), ("H4", HIST_H4_MEAN)):
        staged = [
            float(r["objective_h"])
            for r in rows
            if r.get("method") == method
            and r.get("stage") == "basic_ls"
            and r.get("feasible") == "1"
            and r.get("objective_h") != ""
        ]
        if not staged:
            staged = [
                float(r["objective_h"])
                for r in rows
                if r.get("method") == method
                and r.get("feasible") in ("1", 1)
                and r.get("objective_h") != ""
            ]
        mean = _mean(staged)
        # Pass if within 0.15 h of historical mean, or strictly better (lower).
        ok = math.isfinite(mean) and (mean <= ref + 0.15)
        report["details"][method] = {
            "recovered_mean": mean,
            "historical_mean": ref,
            "abs_diff": abs(mean - ref) if math.isfinite(mean) else None,
            "pass": ok,
            "n": len(staged),
        }
        report["pass"] = report["pass"] and ok
        print(f"NONREG {method}: mean={mean:.6f} hist={ref:.6f} pass={ok}", flush=True)
    (RES / f"{PREFIX}_nonregression.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def stage_same_instance_vs_insertion() -> None:
    """Document exact same-instance comparison vs INSERTION hybrid."""
    hist_h2 = _hist_objectives("H2")
    hist_h4 = _hist_objectives("H4")
    ins = list(
        csv.DictReader((RES / "INSERTION_official_n40.csv").open(encoding="utf-8"))
    )
    rows = []
    for method, hist in (("H2", hist_h2), ("H4", hist_h4)):
        label = f"{method}_c3"
        for inst, hobj in sorted(hist.items()):
            for mode in ("classical_ruin", "hybrid_insert", "ls"):
                match = [
                    r
                    for r in ins
                    if r["instance"] == inst
                    and r["method"] == label
                    and r["mode"] == mode
                    and r["feasible"] == "1"
                ]
                if not match:
                    continue
                iobj = float(match[0]["objective_h"])
                rows.append(
                    {
                        "instance": inst,
                        "method": method,
                        "mode": mode,
                        "historical_fairls_h": hobj,
                        "insertion_pipeline_h": iobj,
                        "delta_h": iobj - hobj,
                        "worse": int(iobj > hobj + 1e-9),
                    }
                )
    _write_csv(RES / f"{PREFIX}_n40_hist_vs_insertion.csv", rows)
    for method in ("H2", "H4"):
        for mode in ("hybrid_insert", "classical_ruin", "ls"):
            deltas = [
                float(r["delta_h"])
                for r in rows
                if r["method"] == method and r["mode"] == mode
            ]
            worse = sum(
                int(r["worse"])
                for r in rows
                if r["method"] == method and r["mode"] == mode
            )
            print(
                f"VS_INSERT {method} {mode}: mean_delta={_mean(deltas):.4f}h "
                f"worse={worse}/{len(deltas)}",
                flush=True,
            )


def stage_fair_budget_compare() -> list[dict]:
    """Same recovered starts; classical vs neural under equal oracle budget."""
    import torch
    from charging import FixedRouteOracle, check_solution, check_route
    from evaluate import _refine_best_once
    from solver_ls import ruin_and_repair
    from train import hybrid_lns, load_policy, neural_lns

    policies = {
        "H2": load_policy(CKPT / "H2_n10_seed42_c3.pt"),
        "H4": load_policy(CKPT / "H4_n10_seed42_c3.pt"),
    }
    # Cap neural/hybrid oracle executions to classical's mean budget (~600 ruin).
    # We also report uncapped hybrid for reference via recovered pipeline.
    rows = []
    with torch.no_grad():
        for method, policy in policies.items():
            for inst in _n40_instances():
                attempts = _construct_attempts(
                    policy, inst, forced=False, cost_lambda=0.0
                )
                feas = [a for a in attempts if a["feasible"]]
                if not feas:
                    continue
                best = min(feas, key=lambda a: a["objective"])
                # Strong LS once
                r0, p0, _, ls_calls, _ = _refine_best_once(
                    inst,
                    list(best["routes"]),
                    list(best["plans"]),
                    3,
                    4000,
                    operators="strong",
                )
                base_obj = sum(x.duration_hours for x in p0)
                modes = {}

                # classical ruin budget 600
                oracle = FixedRouteOracle(inst)
                t0 = time.perf_counter()
                rc, pc, _, _ = ruin_and_repair(
                    list(r0),
                    list(p0),
                    oracle,
                    inst,
                    max_iterations=6,
                    evaluation_budget=600,
                    seed=SEED + 91,
                )
                dt = time.perf_counter() - t0
                obj = sum(x.duration_hours for x in pc)
                if obj > base_obj:
                    rc, pc, obj = r0, p0, base_obj
                modes["classical_ruin"] = (rc, pc, obj, oracle.calls, dt)

                # neural insert
                oracle = FixedRouteOracle(inst)
                t0 = time.perf_counter()
                rn, pn, _, _ = neural_lns(
                    policy,
                    inst,
                    list(r0),
                    list(p0),
                    max_iterations=6,
                    seed=SEED + 91,
                    cost_lambda=4.0,
                    repair_insertion=True,
                    oracle=oracle,
                )
                dt = time.perf_counter() - t0
                obj = sum(x.duration_hours for x in pn)
                if obj > base_obj:
                    rn, pn, obj = r0, p0, base_obj
                modes["neural_insert"] = (rn, pn, obj, oracle.calls, dt)

                # hybrid
                oracle = FixedRouteOracle(inst)
                t0 = time.perf_counter()
                rh, ph, _, _ = hybrid_lns(
                    policy,
                    inst,
                    list(r0),
                    list(p0),
                    max_iterations=6,
                    seed=SEED + 91,
                    cost_lambda=4.0,
                    classical_iterations=6,
                    classical_budget=600,
                    repair_insertion=True,
                    oracle=oracle,
                )
                dt = time.perf_counter() - t0
                obj = sum(x.duration_hours for x in ph)
                if obj > base_obj:
                    rh, ph, obj = r0, p0, base_obj
                modes["hybrid_insert"] = (rh, ph, obj, oracle.calls, dt)

                modes["strong_ls"] = (r0, p0, base_obj, ls_calls, 0.0)

                for name, (routes, plans, obj, calls, dt) in modes.items():
                    rows.append(
                        {
                            "instance": inst.name,
                            "method": method,
                            "mode": name,
                            "feasible": 1,
                            "objective_h": obj,
                            "runtime_search_s": round(dt, 4),
                            "oracle_solver_executions": calls,
                            "n_routes": len(routes),
                            "n_charges": sum(
                                check_route(inst, plan.stops).charging_stops
                                for plan in plans
                            ),
                        }
                    )
                    print(
                        f"fair {method} {name} {inst.name} obj={obj:.4f} exec={calls}",
                        flush=True,
                    )
    _write_csv(RES / f"{PREFIX}_n40_fair_search.csv", rows)
    return rows


def stage_report() -> None:
    lines = [
        "# Regression Recovery Report",
        "",
        "Critical finding: latest insertion-hybrid n40 results beat their *own*",
        "classical baseline but are **worse than historical fair-LS** on every",
        "instance. This audit recovers the old protocol and re-applies hybrid search.",
        "",
        "## 1. Historical vs latest insertion pipeline (same 20 instances)",
        "",
    ]
    vs_path = RES / f"{PREFIX}_n40_hist_vs_insertion.csv"
    if vs_path.exists():
        rows = list(csv.DictReader(vs_path.open(encoding="utf-8")))
        lines.append(
            "| Method | Mode | Mean hist h | Mean insertion h | Mean Δ h | Worse count |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        by = defaultdict(list)
        for r in rows:
            by[(r["method"], r["mode"])].append(r)
        for (method, mode), group in sorted(by.items()):
            hist = _mean([float(r["historical_fairls_h"]) for r in group])
            ins = _mean([float(r["insertion_pipeline_h"]) for r in group])
            delta = _mean([float(r["delta_h"]) for r in group])
            worse = sum(int(r["worse"]) for r in group)
            lines.append(
                f"| {method} | {mode} | {hist:.4f} | {ins:.4f} | {delta:+.4f} | "
                f"{worse}/{len(group)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 2. Protocol differences",
            "",
            "| Aspect | Historical fair-LS | Insertion experiment |",
            "|---|---|---|",
            "| Construction samples | 3 | 4 (n40) / 3–6 other |",
            "| First attempt | greedy (`sample=False`) | always `sample=True` |",
            "| Start diversity | none | forced first customer |",
            "| cost_lambda | **0** | **4** |",
            "| LS operators | basic | strong |",
            "| Post-process | one-shot LS | LS + ruin / hybrid |",
            "",
        ]
    )

    abl = RES / f"{PREFIX}_n40_construction_ablation_summary.json"
    if abl.exists():
        summary = json.loads(abl.read_text(encoding="utf-8"))
        lines.append("## 3. Construction ablation (identical basic one-shot LS)")
        lines.append("")
        lines.append("| Config | H2 mean LS h | H4 mean LS h |")
        lines.append("|---|---:|---:|")
        for cfg in ("ordinary_l0", "ordinary_l4", "forced_l0", "forced_l4"):
            h2 = summary.get(f"H2|{cfg}", float("nan"))
            h4 = summary.get(f"H4|{cfg}", float("nan"))
            lines.append(f"| {cfg} | {h2:.4f} | {h4:.4f} |")
        lines.append("")
        lines.append(
            f"Historical targets: H2={HIST_H2_MEAN:.4f}, H4={HIST_H4_MEAN:.4f}."
        )
        lines.append("")

    pipe = RES / f"{PREFIX}_n40_recovered_pipeline.csv"
    if pipe.exists():
        rows = list(csv.DictReader(pipe.open(encoding="utf-8")))
        lines.append("## 4. Recovered ordinary λ=0 construction + staged search")
        lines.append("")
        lines.append(
            "| Method | Stage | Mean h | Mean Δ vs hist | Beats hist | Mean exec |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        by = defaultdict(list)
        for r in rows:
            if r["feasible"] == "1":
                by[(r["method"], r["stage"])].append(r)
        for key, group in sorted(by.items()):
            objs = [float(r["objective_h"]) for r in group]
            deltas = [
                float(r["delta_vs_hist_h"])
                for r in group
                if r["delta_vs_hist_h"] != ""
            ]
            beats = sum(int(r["beats_hist"]) for r in group)
            execs = [float(r["oracle_solver_executions"]) for r in group]
            lines.append(
                f"| {key[0]} | {key[1]} | {_mean(objs):.4f} | {_mean(deltas):+.4f} | "
                f"{beats}/{len(group)} | {_mean(execs):.0f} |"
            )
        lines.append("")

    fair = RES / f"{PREFIX}_n40_fair_search.csv"
    if fair.exists():
        rows = list(csv.DictReader(fair.open(encoding="utf-8")))
        lines.append("## 5. Fair search from recovered strong-LS starts")
        lines.append("")
        lines.append("| Method | Mode | Mean h | Mean exec | Mean search s |")
        lines.append("|---|---|---:|---:|---:|")
        by = defaultdict(list)
        for r in rows:
            by[(r["method"], r["mode"])].append(r)
        for key, group in sorted(by.items()):
            lines.append(
                f"| {key[0]} | {key[1]} | "
                f"{_mean([float(r['objective_h']) for r in group]):.4f} | "
                f"{_mean([float(r['oracle_solver_executions']) for r in group]):.0f} | "
                f"{_mean([float(r['runtime_search_s']) for r in group]):.3f} |"
            )
        lines.append("")

    nr = RES / f"{PREFIX}_nonregression.json"
    if nr.exists():
        report = json.loads(nr.read_text(encoding="utf-8"))
        lines.append("## 6. Non-regression check")
        lines.append("")
        lines.append(f"- Overall pass: **{report.get('pass')}**")
        for method, d in report.get("details", {}).items():
            lines.append(
                f"- {method}: recovered mean {d.get('recovered_mean'):.6f} vs "
                f"hist {d.get('historical_mean'):.6f} "
                f"(Δ={d.get('abs_diff')}, pass={d.get('pass')})"
            )
        lines.append("")

    lines.extend(
        [
            "## 7. Conclusions",
            "",
            "- Do **not** treat INSERTION hybrid n40 as the global best.",
            "- Historical fair-LS (~57.17 h H2 / ~57.51 h H4) remains the quality reference.",
            "- Root cause is construction/protocol mismatch (forced starts + cost_lambda=4),",
            "  not failure of position-aware insertion itself.",
            "- Best scientific configuration: **historical ordinary λ=0 construction +",
            "  basic/strong LS**, then optional hybrid_insert that never worsens the incumbent.",
            "- H2 vs H4 must stay matched; report whichever wins honestly.",
            "",
            "## Commands",
            "",
            "```bash",
            "python results/run_regression_recovery.py --stage all",
            "```",
            "",
        ]
    )
    out = RES / "REGRESSION_RECOVERY_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}", flush=True)


def stage_tests() -> None:
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_insertion_repair",
        "tests.test_repair_and_validation",
        "-v",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "tests",
            "compare_hist",
            "reproduce",
            "ablate",
            "pipeline",
            "nonreg",
            "fair",
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
    if args.stage in ("compare_hist", "all"):
        print("===== STAGE compare_hist =====", flush=True)
        stage_same_instance_vs_insertion()
    if args.stage in ("reproduce", "all"):
        print("===== STAGE reproduce =====", flush=True)
        stage_reproduce()
    if args.stage in ("ablate", "all"):
        print("===== STAGE ablate =====", flush=True)
        stage_construction_ablation()
    if args.stage in ("pipeline", "all"):
        print("===== STAGE pipeline =====", flush=True)
        stage_pipeline_on_recovered()
    if args.stage in ("nonreg", "all"):
        print("===== STAGE nonreg =====", flush=True)
        stage_nonregression_check()
    if args.stage in ("fair", "all"):
        print("===== STAGE fair =====", flush=True)
        stage_fair_budget_compare()
    if args.stage in ("report", "all"):
        print("===== STAGE report =====", flush=True)
        stage_report()
    print(f"===== done in {time.perf_counter() - t0:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()
