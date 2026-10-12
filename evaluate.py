"""Phase-1 inspection, numerical sanity checks, and baseline experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import permutations
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


def run_compare(
    data_dir: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    size: int,
    limit: int,
    samples: int,
    seed: int,
    checkpoint_size: int | None,
) -> None:
    """Evaluate H0-H4 with the same number of constructed solutions."""
    import torch
    from train import decode, load_policy

    selected = [
        path
        for path in instance_files(data_dir)
        if len(parse_instance(path).customers) == size
    ][:limit]
    trained_size = checkpoint_size or size
    policies = {
        method: load_policy(checkpoints_dir / f"{method}_n{trained_size}_seed{seed}.pt")
        for method in ("H1", "H2", "H3", "H4")
    }
    rows = []
    for path in selected:
        instance = parse_instance(path)
        h0 = randomized_greedy(
            instance,
            restarts=samples,
            seed=seed,
            local_search_passes=0,
            local_search_budget=0,
        )
        rows.append(
            {
                "instance": instance.name,
                "customers": size,
                "method": "H0",
                "seed": seed,
                "samples": samples,
                "feasible": int(h0.feasible),
                "objective_h": h0.objective_hours if h0.feasible else "",
                "routes": h0.routes if h0.feasible else "",
                "charging_stops": h0.charging_stops if h0.feasible else "",
                "charging_h": h0.charging_hours if h0.feasible else "",
                "runtime_s": h0.runtime_seconds,
                "oracle_calls": h0.oracle_calls,
                "reason": h0.reason,
            }
        )
        for method, policy in policies.items():
            attempts = []
            with torch.no_grad():
                for sample_index in range(samples):
                    attempts.append(
                        decode(
                            policy,
                            instance,
                            sample=sample_index > 0,
                            seed=seed * 1000 + sample_index,
                        )
                    )
            feasible = [attempt for attempt in attempts if attempt["feasible"]]
            best = min(feasible, key=lambda attempt: attempt["objective"]) if feasible else None
            charging_stops = charging_hours = 0
            if best is not None:
                for plan in best["plans"]:
                    check = check_route(instance, plan.stops)
                    charging_stops += check.charging_stops
                    charging_hours += check.charging_hours
            rows.append(
                {
                    "instance": instance.name,
                    "customers": size,
                    "method": method,
                    "seed": seed,
                    "samples": samples,
                    "feasible": int(best is not None),
                    "objective_h": best["objective"] if best else "",
                    "routes": len(best["routes"]) if best else "",
                    "charging_stops": charging_stops if best else "",
                    "charging_h": charging_hours if best else "",
                    "runtime_s": sum(attempt["runtime"] for attempt in attempts),
                    "oracle_calls": sum(attempt["oracle_calls"] for attempt in attempts),
                    "reason": "" if best else attempts[-1]["reason"],
                }
            )
        print(f"evaluated {instance.name}")

    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_from{trained_size}" if trained_size != size else ""
    output = results_dir / f"compare_size{size}{suffix}_seed{seed}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for method in ("H0", "H1", "H2", "H3", "H4"):
        method_rows = [row for row in rows if row["method"] == method]
        feasible = [row for row in method_rows if row["feasible"]]
        objectives = [float(row["objective_h"]) for row in feasible]
        print(
            f"{method}: feasible={len(feasible)}/{len(method_rows)}, "
            f"mean={statistics.mean(objectives):.6f}, "
            f"median={statistics.median(objectives):.6f}, "
            f"time={sum(float(row['runtime_s']) for row in method_rows):.3f}s"
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
) -> None:
    """Compare realized learned charging with hybrid and post-optimized routes."""
    import torch
    from train import decode, decode_direct, load_direct_policy, load_policy

    routing = load_policy(checkpoints_dir / f"H4_n{size}_seed{seed}.pt")
    charging = load_direct_policy(
        checkpoints_dir / f"CHARGE_n{size}_seed{seed}.pt"
    )
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
    output = results_dir / f"direct_charge_size{size}_seed{seed}.csv"
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


def make_plots(results_dir: Path, seed: int) -> None:
    """Create compact learning and benchmark figures from executed CSVs."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for method in ("H1", "H2", "H3", "H4"):
        path = results_dir / f"training_{method}_n10_seed{seed}.csv"
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
    learning_path = results_dir / "learning_curves.png"
    fig.savefig(learning_path, dpi=160)
    plt.close(fig)

    compare = []
    for candidate_seed in (seed, seed + 1):
        compare_path = results_dir / f"compare_size10_seed{candidate_seed}.csv"
        if compare_path.exists():
            with compare_path.open(newline="", encoding="utf-8") as handle:
                compare.extend(csv.DictReader(handle))
    methods = ("H0", "H1", "H2", "H3", "H4")
    means = [
        statistics.mean(
            float(row["objective_h"])
            for row in compare
            if row["method"] == method and int(row["feasible"])
        )
        for method in methods
    ]
    fig, axis = plt.subplots(figsize=(6, 3.5))
    bars = axis.bar(methods, means)
    axis.set_ylabel("Mean verified route duration (h)")
    axis.set_title("Controlled 10-construction budget (H0 without local search)")
    axis.bar_label(bars, fmt="%.2f")
    fig.tight_layout()
    compare_figure = results_dir / "h0_h4_comparison.png"
    fig.savefig(compare_figure, dpi=160)
    plt.close(fig)
    print(f"Wrote {learning_path} and {compare_figure}")


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
    parser.add_argument("--seed", type=int, default=42)
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
        )
    else:
        make_plots(args.results, args.seed)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
