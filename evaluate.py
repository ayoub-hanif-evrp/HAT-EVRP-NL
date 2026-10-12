"""Phase-1 inspection, numerical sanity checks, and baseline experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import permutations
from pathlib import Path
import platform
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
        ours["energy_matrix"][node][helper] > instance.battery_wh
        for node in range(len(instance.nodes))
        if node != instance.depot
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
        result = randomized_greedy(instance, restarts=restarts, seed=seed)
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
    output = results_dir / f"baseline_size{size}_seed{seed}.csv"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inspect", "sanity", "baseline"), required=True)
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--instance", default="tc0c10s2cf1")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "inspect":
        inspect_dataset(args.data, args.results)
    elif args.mode == "sanity":
        run_sanity(args.data, args.instance)
    else:
        run_baselines(
            args.data, args.results, args.size, args.limit, args.restarts, args.seed
        )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
