"""Phase-1 inspection, numerical sanity checks, and baseline experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import permutations
import math
from pathlib import Path
import platform
import statistics
import sys

from charging import (
    FixedRouteOracle,
    charging_time,
    check_route,
    cumulative_charging_time,
    to_frvcpy,
)
from data import instance_files, parse_instance
from solver import randomized_greedy


def inspect_dataset(data_dir: Path, results_dir: Path) -> None:
    files = instance_files(data_dir)
    if not files:
        raise RuntimeError(f"No XML files found under {data_dir}")
    rows = []
    group_counts: dict[int, int] = {}
    for path in files:
        instance = parse_instance(path)
        n = len(instance.customers)
        group_counts[n] = group_counts.get(n, 0) + 1
        xs = [node.x for node in instance.nodes]
        ys = [node.y for node in instance.nodes]
        rows.append(
            {
                "instance": instance.name,
                "customers": n,
                "stations": len(instance.stations),
                "station_types": ",".join(sorted(set(
                    instance.nodes[s].station_type for s in instance.stations
                ))),
                "battery_wh": instance.battery_wh,
                "route_limit_h": instance.max_route_hours,
                "x_min_km": min(xs),
                "x_max_km": max(xs),
                "y_min_km": min(ys),
                "y_max_km": max(ys),
                "average_service_h": (
                    sum(instance.service_times[c] for c in instance.customers) / n
                ),
            }
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / "dataset_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"XML instances: {len(files)}; size groups: {dict(sorted(group_counts.items()))}")
    print(
        "Semantics: Euclidean km (unrounded double precision), 40 km/h, "
        "125 Wh/km, Q=16000 Wh, T_max=10 h, service=0.5 h/customer"
    )
    print(f"Wrote {output}")


def run_sanity(data_dir: Path, instance_name: str) -> None:
    matches = list(data_dir.rglob(f"{instance_name}.xml"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {instance_name}, found {len(matches)}")
    instance = parse_instance(matches[0])

    for curve in instance.curves.values():
        assert abs(cumulative_charging_time(curve, 0.0)) < 1e-12
        assert abs(
            cumulative_charging_time(curve, instance.battery_wh) - curve.times[-1]
        ) < 1e-12
        for energy in (0.0, 0.25 * instance.battery_wh, instance.battery_wh):
            assert abs(charging_time(curve, energy, energy)) < 1e-12

    # Verify our matrices and process times against frvcpy's own translator.
    from frvcpy.translator import translate

    translated = translate(str(instance.path), depot_charging=False)
    ours = to_frvcpy(instance)
    assert translated["max_q"] == ours["max_q"]
    assert translated["t_max"] == ours["t_max"]
    assert translated["process_times"] == ours["process_times"][:-1]
    for key in ("time_matrix", "energy_matrix"):
        max_error = max(
            abs(a - b)
            for row_a, row_b in zip(translated[key], ours[key][:-1])
            for a, b in zip(row_a, row_b[:-1])
        )
        # math.hypot and sqrt(dx**2 + dy**2) may differ by a few ULPs.
        assert max_error < 1e-9, (key, max_error)
    helper = len(instance.nodes)
    assert ours["energy_matrix"][instance.depot][helper] == 0.0
    assert all(
        ours["energy_matrix"][node][helper]
        == instance.energy_wh[node][instance.depot]
        for node in range(len(instance.nodes))
    )

    oracle = FixedRouteOracle(instance)
    demonstrated = None
    first_feasible = None
    for length in (1, 2, 3):
        for sequence in permutations(instance.customers, length):
            plan = oracle.solve(sequence)
            if not plan.feasible:
                continue
            check = check_route(instance, plan.stops, sequence)
            assert check.feasible
            assert abs(check.duration_hours - plan.duration_hours) < 1e-7
            if first_feasible is None:
                first_feasible = (sequence, plan, check)
            if check.charging_stops:
                demonstrated = (sequence, plan, check)
                break
        if demonstrated is not None:
            break
    if demonstrated is None:
        assert first_feasible is not None
        demonstrated = first_feasible

    sequence, plan, check = demonstrated
    tighter = replace(instance, max_route_hours=check.duration_hours - 1e-5)
    over_limit = check_route(tighter, plan.stops, sequence)
    assert not over_limit.feasible and "exceeds" in over_limit.reason

    print(f"Sanity instance: {instance.name}")
    print(
        f"Verified route customers={sequence}, duration={check.duration_hours:.9f} h, "
        f"charging={check.charging_hours:.9f} h, charging stops={check.charging_stops}, "
        f"end SOC={check.end_soc_wh:.3f} Wh"
    )
    print(f"Concrete route: {plan.stops}")
    print(f"Independent route-limit rejection: {over_limit.reason}")
    print(f"frvcpy calls: {oracle.calls}; all sanity checks passed")


def run_baselines(
    data_dir: Path,
    results_dir: Path,
    size: int,
    limit: int,
    restarts: int,
    seed: int,
    local_passes: int,
    local_budget: int,
) -> None:
    selected = [
        path
        for path in instance_files(data_dir)
        if len(parse_instance(path).customers) == size
    ][:limit]
    if not selected:
        raise RuntimeError(f"No {size}-customer instances found")

    rows = []
    for path in selected:
        instance = parse_instance(path)
        result = randomized_greedy(
            instance,
            restarts=restarts,
            seed=seed,
            local_search_passes=local_passes,
            local_search_budget=local_budget,
        )
        row = {
            "instance": instance.name,
            "customers": size,
            "seed": seed,
            "restarts": restarts,
            "feasible": int(result.feasible),
            "objective_h": result.objective_hours if result.feasible else "",
            "routes": result.routes if result.feasible else "",
            "charging_stops": result.charging_stops if result.feasible else "",
            "charging_h": result.charging_hours if result.feasible else "",
            "runtime_s": result.runtime_seconds,
            "local_search_s": result.local_search_seconds,
            "local_improvements": result.local_improvements,
            "oracle_calls": result.oracle_calls,
            "reason": result.reason,
            "customer_routes": repr(result.customer_routes),
        }
        rows.append(row)
        status = (
            f"objective={result.objective_hours:.6f} h, routes={result.routes}, "
            f"charges={result.charging_stops}"
            if result.feasible
            else f"FAILED: {result.reason}"
        )
        print(
            f"{instance.name}: {status}, time={result.runtime_seconds:.3f} s, "
            f"oracle calls={result.oracle_calls}"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / f"h0_size{size}_seed{seed}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    feasible_rows = [row for row in rows if row["feasible"]]
    mean = (
        sum(float(row["objective_h"]) for row in feasible_rows) / len(feasible_rows)
        if feasible_rows
        else float("nan")
    )
    print(
        f"Summary: feasible={len(feasible_rows)}/{len(rows)}, "
        f"mean feasible objective={mean:.6f} h"
    )
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"Wrote {output}")


def _append_method_row(
    rows: list[dict],
    instance,
    size: int,
    method: str,
    seed: int,
    version: str,
    samples: int,
    feasible: bool,
    objective,
    routes,
    charging_stops,
    charging_hours,
    runtime_s,
    oracle_calls,
    reason: str,
    unique_constructions: int | str = "",
    objective_lit="",
    cost_lambda: float | str = "",
) -> None:
    from charging import literature_compatible_objective_hours

    lit = objective_lit
    if feasible and lit == "" and objective != "":
        lit = literature_compatible_objective_hours(instance, float(objective))
    rows.append(
        {
            "instance": instance.name,
            "customers": size,
            "method": method,
            "seed": seed,
            "version": version,
            "samples": samples,
            "feasible": int(feasible),
            "objective_h": objective if feasible else "",
            "objective_lit_h": lit if feasible else "",
            "routes": routes if feasible else "",
            "charging_stops": charging_stops if feasible else "",
            "charging_h": charging_hours if feasible else "",
            "runtime_s": runtime_s,
            "oracle_calls": oracle_calls,
            "unique_constructions": unique_constructions,
            "cost_lambda": cost_lambda,
            "reason": reason,
        }
    )


def _refine_best_once(
    instance,
    customer_routes: list[tuple[int, ...]],
    plans: list,
    local_passes: int,
    local_budget: int,
    operators: str = "basic",
):
    """Apply identical one-shot local search used by H0_strong after best-of-N."""
    from charging import FixedRouteOracle
    from solver import local_search

    oracle = FixedRouteOracle(instance)
    refined_routes, refined_plans, improvements = local_search(
        list(customer_routes),
        list(plans),
        oracle,
        max_passes=local_passes,
        evaluation_budget=local_budget,
        operators=operators,
    )
    return refined_routes, refined_plans, improvements, oracle.calls


def _classical_until_time(
    instance,
    seed: int,
    time_budget_s: float,
    local_passes: int,
    local_budget: int,
):
    """Spend a measured wall-clock budget on classical search.

    Generates genuinely seeded randomized constructions (only the first trial
    may be deterministic). Tracks unique customer-route constructions so that
    duplicated identical trials are not silently treated as independent search.
    """
    import time
    from dataclasses import replace

    from charging import FixedRouteOracle, check_solution
    from solver import local_search

    start = time.perf_counter()
    best_routes = None
    best_plans = None
    best_obj = float("inf")
    restarts = 0
    oracle_calls = 0
    last_fail = ""
    unique_constructions: set[tuple[tuple[int, ...], ...]] = set()

    # Reserve a fraction of the measured neural budget for one final LS.
    construct_deadline = (
        start + time_budget_s * 0.8 if local_passes > 0 else start + time_budget_s
    )
    while time.perf_counter() < construct_deadline and restarts < 5000:
        # Trial 0: one optional deterministic greedy. Later trials: RCL sampling
        # with independent seeds so seed changes actually change actions.
        result = randomized_greedy(
            instance,
            restarts=1,
            seed=seed + 1009 * restarts + 17,
            local_search_passes=0,
            local_search_budget=0,
            deterministic_first=(restarts == 0),
        )
        restarts += 1
        oracle_calls += result.oracle_calls
        if result.feasible:
            unique_constructions.add(result.customer_routes)
            if result.objective_hours < best_obj:
                best_obj = result.objective_hours
                best_routes = list(result.customer_routes)
                best_plans = list(result.plans)
        else:
            last_fail = result.reason
        if restarts == 1 and time.perf_counter() >= construct_deadline:
            break

    if best_routes is None:
        elapsed = time.perf_counter() - start
        failed = randomized_greedy(
            instance,
            restarts=1,
            seed=seed,
            local_search_passes=0,
            local_search_budget=0,
            deterministic_first=True,
        )
        return (
            replace(
                failed,
                feasible=False,
                reason=last_fail or failed.reason,
                runtime_seconds=elapsed,
                oracle_calls=oracle_calls,
                unique_constructions=len(unique_constructions),
            ),
            restarts,
            elapsed,
            len(unique_constructions),
        )

    improvements = 0
    if local_passes > 0:
        oracle = FixedRouteOracle(instance)
        best_routes, best_plans, improvements = local_search(
            best_routes,
            best_plans,
            oracle,
            max_passes=local_passes,
            evaluation_budget=local_budget,
        )
        oracle_calls += oracle.calls

    elapsed = time.perf_counter() - start
    valid, reason = check_solution(instance, best_plans)
    if not valid:
        failed = randomized_greedy(
            instance,
            restarts=1,
            seed=seed,
            local_search_passes=0,
            local_search_budget=0,
            deterministic_first=True,
        )
        return (
            replace(
                failed,
                feasible=False,
                reason=reason,
                runtime_seconds=elapsed,
                oracle_calls=oracle_calls,
                unique_constructions=len(unique_constructions),
            ),
            restarts,
            elapsed,
            len(unique_constructions),
        )
    charging_stops = charging_hours = 0
    for plan in best_plans:
        check = check_route(instance, plan.stops)
        charging_stops += check.charging_stops
        charging_hours += check.charging_hours
    objective = sum(plan.duration_hours for plan in best_plans)
    template = randomized_greedy(
        instance,
        restarts=1,
        seed=seed,
        local_search_passes=0,
        local_search_budget=0,
        deterministic_first=True,
    )
    return (
        replace(
            template,
            feasible=True,
            objective_hours=objective,
            customer_routes=tuple(best_routes),
            plans=tuple(best_plans),
            routes=len(best_routes),
            charging_stops=charging_stops,
            charging_hours=charging_hours,
            oracle_calls=oracle_calls,
            runtime_seconds=elapsed,
            local_search_seconds=0.0,
            local_improvements=improvements,
            reason="",
            unique_constructions=len(unique_constructions),
        ),
        restarts,
        elapsed,
        len(unique_constructions),
    )


def run_compare(
    data_dir: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    size: int,
    limit: int,
    samples: int,
    seed: int,
    checkpoint_size: int | None,
    version: str = "c3",
    h0_local_passes: int = 3,
    h0_local_budget: int = 4000,
    neural_local_passes: int = 0,
    neural_local_budget: int = 0,
    wallclock: bool = False,
    tag: str = "",
    fair_ls: bool = False,
    cost_lambda: float = 0.0,
    methods: tuple[str, ...] = ("H1", "H2", "H3", "H4"),
) -> None:
    """Evaluate H0-H4 with equal construction budgets and optional fair one-shot LS.

    Fair LS protocol (``fair_ls=True``):
    1. Generate ``samples`` constructions for every method WITHOUT local search.
    2. Keep the best feasible construction.
    3. Apply identical ``local_search`` once with the same passes/budget.
    Previous Cycle-3 ``eqls`` runs applied LS to every neural sample and must
    not be treated as equal-effort comparisons.

    ``cost_lambda`` enables optional oracle-aware pointer scoring at inference
    (see ``train.decode``). Default 0 preserves Cycle-3 behavior.
    """
    import time

    import torch
    from charging import check_solution
    from train import decode, load_policy

    selected = [
        path
        for path in instance_files(data_dir)
        if len(parse_instance(path).customers) == size
    ][:limit]
    trained_size = checkpoint_size or size
    policies = {}
    for method in methods:
        path = checkpoints_dir / f"{method}_n{trained_size}_seed{seed}_{version}.pt"
        if not path.exists():
            path = checkpoints_dir / f"{method}_n{trained_size}_seed{seed}_c2.pt"
        if not path.exists():
            path = checkpoints_dir / f"{method}_n{trained_size}_seed{seed}.pt"
        policies[method] = load_policy(path)

    # Fair LS always uses the H0_strong budget; ignore per-sample neural LS.
    refine_passes = h0_local_passes if fair_ls else 0
    refine_budget = h0_local_budget if fair_ls else 0
    if neural_local_passes > 0 and not fair_ls:
        # Preserve legacy (unfair) per-sample LS only when explicitly requested
        # without --fair-ls; tag keeps those CSVs separate.
        pass

    rows = []
    for path in selected:
        instance = parse_instance(path)
        neural_raw: dict[str, list] = {}
        with torch.no_grad():
            for method, policy in policies.items():
                attempts = []
                for sample_index in range(samples):
                    # Construction only — never LS inside each sample for fair_ls.
                    attempts.append(
                        decode(
                            policy,
                            instance,
                            sample=sample_index > 0,
                            seed=seed * 1000 + sample_index,
                            local_passes=0 if fair_ls else neural_local_passes,
                            local_budget=0 if fair_ls else neural_local_budget,
                            cost_lambda=cost_lambda,
                        )
                    )
                neural_raw[method] = attempts

        neural_results = {}
        for method, attempts in neural_raw.items():
            construct_time = sum(attempt["runtime"] for attempt in attempts)
            construct_oracle = sum(attempt["oracle_calls"] for attempt in attempts)
            feasible = [attempt for attempt in attempts if attempt["feasible"]]
            if not feasible:
                neural_results[method] = {
                    "feasible": False,
                    "objective": "",
                    "routes": "",
                    "charging_stops": "",
                    "charging_h": "",
                    "runtime": construct_time,
                    "oracle_calls": construct_oracle,
                    "reason": attempts[-1]["reason"],
                    "samples": samples,
                }
                continue
            best = min(feasible, key=lambda attempt: attempt["objective"])
            routes = list(best["routes"])
            plans = list(best["plans"])
            refine_oracle = 0
            refine_time = 0.0
            if fair_ls and refine_passes > 0:
                refine_start = time.perf_counter()
                routes, plans, _, refine_oracle = _refine_best_once(
                    instance, routes, plans, refine_passes, refine_budget
                )
                refine_time = time.perf_counter() - refine_start
                valid, reason = check_solution(instance, plans)
                objective = (
                    sum(plan.duration_hours for plan in plans) if valid else float("inf")
                )
                feasible_flag = valid
                fail_reason = "" if valid else reason
            else:
                objective = best["objective"]
                feasible_flag = True
                fail_reason = ""
            charging_stops = charging_hours = 0
            if feasible_flag:
                for plan in plans:
                    check = check_route(instance, plan.stops)
                    charging_stops += check.charging_stops
                    charging_hours += check.charging_hours
            neural_results[method] = {
                "feasible": feasible_flag,
                "objective": objective if feasible_flag else "",
                "routes": len(routes) if feasible_flag else "",
                "charging_stops": charging_stops if feasible_flag else "",
                "charging_h": charging_hours if feasible_flag else "",
                "runtime": construct_time + refine_time,
                "oracle_calls": construct_oracle + refine_oracle,
                "reason": fail_reason,
                "samples": samples,
            }

        neural_time = max(result["runtime"] for result in neural_results.values())

        h0_unique = strong_unique = ""
        if wallclock:
            # Match measured neural wall time with classical measured wall time.
            h0_equal, h0_restarts, _, h0_unique = _classical_until_time(
                instance, seed, neural_time, local_passes=0, local_budget=0
            )
            h0_strong, strong_restarts, _, strong_unique = _classical_until_time(
                instance,
                seed,
                neural_time,
                local_passes=h0_local_passes,
                local_budget=h0_local_budget,
            )
            h0_samples = h0_restarts
            strong_samples = strong_restarts
        else:
            h0_equal = randomized_greedy(
                instance,
                restarts=samples,
                seed=seed,
                local_search_passes=0,
                local_search_budget=0,
                deterministic_first=True,
            )
            # H0_strong: best-of-N constructions, then ONE local-search pass.
            h0_strong = randomized_greedy(
                instance,
                restarts=samples,
                seed=seed,
                local_search_passes=h0_local_passes,
                local_search_budget=h0_local_budget,
                deterministic_first=True,
            )
            h0_samples = samples
            strong_samples = samples
            h0_unique = h0_equal.unique_constructions
            strong_unique = h0_strong.unique_constructions

        for label, h0, n_samples, n_unique in (
            ("H0", h0_equal, h0_samples, h0_unique),
            ("H0_strong", h0_strong, strong_samples, strong_unique),
        ):
            _append_method_row(
                rows,
                instance,
                size,
                label,
                seed,
                version,
                n_samples,
                h0.feasible,
                h0.objective_hours,
                h0.routes,
                h0.charging_stops,
                h0.charging_hours,
                h0.runtime_seconds,
                h0.oracle_calls,
                h0.reason,
                n_unique,
                cost_lambda=cost_lambda,
            )
        for method, result in neural_results.items():
            _append_method_row(
                rows,
                instance,
                size,
                method,
                seed,
                version,
                result["samples"],
                result["feasible"],
                result["objective"],
                result["routes"],
                result["charging_stops"],
                result["charging_h"],
                result["runtime"],
                result["oracle_calls"],
                result["reason"],
                "",
                cost_lambda=cost_lambda,
            )
        print(
            f"evaluated {instance.name}"
            + (
                f" neural_wall={neural_time:.3f}s"
                f" H0_unique={h0_unique} H0s_unique={strong_unique}"
                if wallclock
                else ""
            )
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_from{trained_size}" if trained_size != size else ""
    extra = tag
    if not extra:
        if wallclock:
            extra = "wallclock_final"
        elif fair_ls:
            extra = "fairls"
        elif neural_local_passes > 0:
            extra = "eqls_legacy"
    suffix = f"{suffix}_{extra}" if extra else suffix
    output = results_dir / f"compare_size{size}{suffix}_seed{seed}_{version}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for method in ("H0", "H0_strong", "H1", "H2", "H3", "H4"):
        method_rows = [row for row in rows if row["method"] == method]
        feasible = [row for row in method_rows if row["feasible"]]
        objectives = [float(row["objective_h"]) for row in feasible]
        oracle_mean = statistics.mean(float(row["oracle_calls"]) for row in method_rows)
        print(
            f"{method}: feasible={len(feasible)}/{len(method_rows)}, "
            f"mean={statistics.mean(objectives):.6f}, "
            f"median={statistics.median(objectives):.6f}, "
            f"time={sum(float(row['runtime_s']) for row in method_rows):.3f}s, "
            f"oracle/inst={oracle_mean:.1f}"
            if objectives
            else f"{method}: feasible=0/{len(method_rows)}"
        )
    print(f"Wrote {output}")


def run_direct_comparison(
    data_dir: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    size: int,
    limit: int,
    samples: int,
    seed: int,
    version: str = "c2",
) -> None:
    """Compare realized learned charging with hybrid and post-optimized routes."""
    import torch
    from train import decode, decode_direct, load_direct_policy, load_policy

    routing_path = checkpoints_dir / f"H4_n{size}_seed{seed}_{version}.pt"
    if not routing_path.exists():
        routing_path = checkpoints_dir / f"H4_n{size}_seed{seed}_c2.pt"
    charging_path = checkpoints_dir / f"CHARGE_n{size}_seed{seed}_{version}.pt"
    if not charging_path.exists():
        charging_path = checkpoints_dir / f"CHARGE_n{size}_seed{seed}_c2.pt"
    routing = load_policy(routing_path)
    charging = load_direct_policy(charging_path)
    selected = [
        path
        for path in instance_files(data_dir)
        if len(parse_instance(path).customers) == size
    ][:limit]
    rows = []
    for path in selected:
        instance = parse_instance(path)
        with torch.no_grad():
            attempts = [
                decode(
                    routing,
                    instance,
                    sample=index > 0,
                    seed=seed * 1000 + index,
                )
                for index in range(samples)
            ]
            feasible = [attempt for attempt in attempts if attempt["feasible"]]
            hybrid = min(feasible, key=lambda attempt: attempt["objective"])
            direct_routes = [
                decode_direct(charging, instance, route) for route in hybrid["routes"]
            ]
        direct_feasible = all(route["feasible"] for route in direct_routes)
        hybrid_charges = hybrid_charge_time = 0
        for plan in hybrid["plans"]:
            check = check_route(instance, plan.stops)
            hybrid_charges += check.charging_stops
            hybrid_charge_time += check.charging_hours
        direct_charges = direct_charge_time = 0
        for result in direct_routes:
            check = check_route(instance, result["stops"])
            direct_charges += check.charging_stops
            direct_charge_time += check.charging_hours
        common = {
            "instance": instance.name,
            "customers": size,
            "seed": seed,
            "samples": samples,
        }
        rows.extend(
            [
                {
                    **common,
                    "method": "hybrid_H4",
                    "feasible": 1,
                    "objective_h": hybrid["objective"],
                    "routes": len(hybrid["routes"]),
                    "charging_stops": hybrid_charges,
                    "charging_h": hybrid_charge_time,
                    "runtime_s": sum(attempt["runtime"] for attempt in attempts),
                    "reason": "",
                },
                {
                    **common,
                    "method": "direct_CHARGE",
                    "feasible": int(direct_feasible),
                    "objective_h": (
                        sum(route["objective"] for route in direct_routes)
                        if direct_feasible
                        else ""
                    ),
                    "routes": len(direct_routes),
                    "charging_stops": direct_charges if direct_feasible else "",
                    "charging_h": direct_charge_time if direct_feasible else "",
                    "runtime_s": sum(route["runtime"] for route in direct_routes),
                    "reason": (
                        ""
                        if direct_feasible
                        else "; ".join(
                            route["reason"]
                            for route in direct_routes
                            if not route["feasible"]
                        )
                    ),
                },
                {
                    **common,
                    "method": "direct_plus_repair",
                    "feasible": 1,
                    "objective_h": hybrid["objective"],
                    "routes": len(hybrid["routes"]),
                    "charging_stops": hybrid_charges,
                    "charging_h": hybrid_charge_time,
                    "runtime_s": (
                        sum(route["runtime"] for route in direct_routes)
                        + sum(attempt["runtime"] for attempt in attempts)
                    ),
                    "reason": "customer orders post-optimized by frvcpy",
                },
            ]
        )
        print(f"evaluated direct charging on {instance.name}")
    output = results_dir / f"direct_charge_size{size}_seed{seed}_{version}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for method in ("hybrid_H4", "direct_CHARGE", "direct_plus_repair"):
        method_rows = [row for row in rows if row["method"] == method]
        feasible = [row for row in method_rows if row["feasible"]]
        objectives = [float(row["objective_h"]) for row in feasible]
        print(
            f"{method}: feasible={len(feasible)}/{len(method_rows)}, "
            + (
                f"mean={statistics.mean(objectives):.6f} h"
                if objectives
                else "mean=NA"
            )
        )
    print(f"Wrote {output}")


def make_plots(results_dir: Path, seed: int, version: str = "c3") -> None:
    """Create compact learning and benchmark figures from executed CSVs."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for method in ("H1", "H2", "H3", "H4"):
        path = results_dir / f"training_{method}_n10_seed{seed}_{version}.csv"
        if not path.exists():
            path = results_dir / f"training_{method}_n10_seed{seed}_c2.csv"
        if not path.exists():
            path = results_dir / f"training_{method}_n10_seed{seed}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["stage"] == "bc"]
        steps = [int(row["step"]) for row in rows]
        axes[0].plot(steps, [float(row["loss"]) for row in rows], label=method)
        axes[1].plot(
            steps,
            [float(row["validation_objective"]) for row in rows],
            label=method,
        )
    axes[0].set(xlabel="BC update", ylabel="Imitation loss")
    axes[1].set(xlabel="BC update", ylabel="Validation duration (h)")
    axes[0].legend(ncol=2)
    axes[1].legend(ncol=2)
    fig.tight_layout()
    learning_path = results_dir / f"learning_curves_{version}.png"
    fig.savefig(learning_path, dpi=160)
    plt.close(fig)

    compare = []
    for candidate_seed in (seed, seed + 1, seed + 2):
        for suffix in (f"_{version}", "_c2", ""):
            compare_path = (
                results_dir / f"compare_size10_seed{candidate_seed}{suffix}.csv"
            )
            if compare_path.exists():
                with compare_path.open(newline="", encoding="utf-8") as handle:
                    compare.extend(csv.DictReader(handle))
                break
    methods = ("H0", "H0_strong", "H1", "H2", "H3", "H4")
    present = [m for m in methods if any(row["method"] == m for row in compare)]
    means = [
        statistics.mean(
            float(row["objective_h"])
            for row in compare
            if row["method"] == method and int(row["feasible"])
        )
        for method in present
    ]
    fig, axis = plt.subplots(figsize=(7, 3.5))
    bars = axis.bar(present, means)
    axis.set_ylabel("Mean verified route duration (h)")
    axis.set_title(f"Comparison {version} (equal samples + strong H0)")
    axis.bar_label(bars, fmt="%.2f")
    fig.tight_layout()
    compare_figure = results_dir / f"h0_h4_comparison_{version}.png"
    fig.savefig(compare_figure, dpi=160)
    plt.close(fig)

    # Paired H4-H2 deltas when available.
    paired_path = results_dir / f"deepsets_paired_{version}.png"
    if compare:
        by_key: dict[tuple[str, str], dict[str, float]] = {}
        for row in compare:
            if not int(row["feasible"]):
                continue
            key = (row["instance"], str(row.get("seed", seed)))
            by_key.setdefault(key, {})[row["method"]] = float(row["objective_h"])
        deltas = [
            values["H4"] - values["H2"]
            for values in by_key.values()
            if "H4" in values and "H2" in values
        ]
        if deltas:
            fig, axis = plt.subplots(figsize=(6, 3.5))
            axis.hist(deltas, bins=12, color="#4c72b0", edgecolor="white")
            axis.axvline(0.0, color="black", linewidth=1)
            axis.set_xlabel("H4 − H2 verified duration (h)")
            axis.set_ylabel("Count")
            axis.set_title("DeepSets contribution (paired)")
            fig.tight_layout()
            fig.savefig(paired_path, dpi=160)
            plt.close(fig)
    print(f"Wrote {learning_path}, {compare_figure}, and {paired_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("inspect", "sanity", "baseline", "compare", "direct", "plots"),
        required=True,
    )
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--instance", default="tc0c10s2cf1")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--local-passes", type=int, default=2)
    parser.add_argument("--local-budget", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-size", type=int)
    parser.add_argument("--version", type=str, default="c3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neural-local-passes", type=int, default=0)
    parser.add_argument("--neural-local-budget", type=int, default=0)
    parser.add_argument("--wallclock", action="store_true")
    parser.add_argument(
        "--fair-ls",
        action="store_true",
        help="Best-of-N constructions then one shared local-search for every method",
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--cost-lambda",
        type=float,
        default=0.0,
        help="Optional oracle-aware pointer score weight (0 = Cycle-3 baseline)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="H1,H2,H3,H4",
        help="Comma-separated neural methods to evaluate",
    )
    args = parser.parse_args()

    if args.mode == "inspect":
        inspect_dataset(args.data, args.results)
    elif args.mode == "sanity":
        run_sanity(args.data, args.instance)
    elif args.mode == "baseline":
        run_baselines(
            args.data,
            args.results,
            args.size,
            args.limit,
            args.restarts,
            args.seed,
            args.local_passes,
            args.local_budget,
        )
    elif args.mode == "compare":
        run_compare(
            args.data,
            args.results,
            args.checkpoints,
            args.size,
            args.limit,
            args.samples,
            args.seed,
            args.checkpoint_size,
            args.version,
            args.local_passes,
            args.local_budget,
            args.neural_local_passes,
            args.neural_local_budget,
            args.wallclock,
            args.tag,
            args.fair_ls,
            args.cost_lambda,
            tuple(method.strip() for method in args.methods.split(",") if method.strip()),
        )
    elif args.mode == "direct":
        run_direct_comparison(
            args.data,
            args.results,
            args.checkpoints,
            args.size,
            args.limit,
            args.samples,
            args.seed,
            args.version,
        )
    else:
        make_plots(args.results, args.seed, args.version)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
