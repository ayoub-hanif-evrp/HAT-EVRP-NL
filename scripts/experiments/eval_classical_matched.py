"""Classical search at the neural protocols' realized solver budgets.

Budgets are the mean frvcpy executions (or mean wall-clock) of P0, P1, and P2
on the official BC/RL matrix, by problem size. The neural checkpoints are not
retrained. Infeasible runs keep a blank objective.
"""

from __future__ import annotations

import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import FixedRouteOracle, check_solution, literature_compatible_objective_hours
from data import instance_files, parse_instance
from solver_ls import local_search, ruin_and_repair

NEURAL = ROOT / "results" / "raw" / "bc_rl_full_v1" / "eval"
OUT = ROOT / "results" / "raw" / "classical_matched_v1" / "official.csv"


def instances_of_size(data_dir: Path, size: int) -> list:
    return [path for path in instance_files(data_dir) if len(parse_instance(path).customers) == size]


def neural_budgets() -> dict[tuple[int, str], dict[str, float]]:
    sums = defaultdict(lambda: [0.0, 0.0, 0])
    for path in NEURAL.glob("H*_seed*.csv"):
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["size"]), row["protocol"])
                sums[key][0] += float(row["oracle_calls"])
                sums[key][1] += float(row["runtime_s"])
                sums[key][2] += 1
    return {
        key: {"executions": calls / n, "seconds": runtime / n}
        for key, (calls, runtime, n) in sums.items()
        if n
    }


def _room(oracle: FixedRouteOracle, limit: int | None, deadline: float | None) -> bool:
    if limit is not None and oracle.calls >= limit:
        return False
    if deadline is not None and time.perf_counter() >= deadline:
        return False
    return True


def greedy_construct(instance, oracle: FixedRouteOracle, limit: int | None, deadline: float | None):
    unassigned = set(instance.customers)
    routes = []
    plans = []
    while unassigned and _room(oracle, limit, deadline):
        sequence: tuple[int, ...] = ()
        current = 0.0
        final = None
        while unassigned and _room(oracle, limit, deadline):
            options = []
            for customer in sorted(unassigned):
                if not _room(oracle, limit, deadline):
                    break
                plan = oracle.solve((*sequence, customer))
                if plan.feasible:
                    options.append((plan.duration_hours - current, customer, plan))
            if not options:
                break
            options.sort(key=lambda item: (item[0], item[1]))
            _delta, customer, final = options[0]
            sequence = (*sequence, customer)
            current = final.duration_hours
            unassigned.remove(customer)
        if not sequence or final is None:
            break
        routes.append(sequence)
        plans.append(final)
    return routes, plans, unassigned


def rcl_construct(instance, oracle, limit, deadline, rng: random.Random, rcl_size: int = 3):
    """Randomized cheapest insertion on the shared oracle."""
    unassigned = set(instance.customers)
    routes = []
    plans = []
    while unassigned and _room(oracle, limit, deadline):
        sequence: tuple[int, ...] = ()
        current = 0.0
        final = None
        while unassigned and _room(oracle, limit, deadline):
            options = []
            for customer in sorted(unassigned):
                if not _room(oracle, limit, deadline):
                    break
                plan = oracle.solve((*sequence, customer))
                if plan.feasible:
                    options.append((plan.duration_hours - current, customer, plan))
            if not options:
                break
            options.sort(key=lambda item: (item[0], item[1]))
            _delta, customer, final = rng.choice(options[:rcl_size])
            sequence = (*sequence, customer)
            current = final.duration_hours
            unassigned.remove(customer)
        if not sequence or final is None:
            break
        routes.append(sequence)
        plans.append(final)
    return routes, plans, unassigned


def _complete(instance, routes) -> bool:
    covered = [customer for route in routes for customer in route]
    return len(covered) == len(instance.customers) and len(set(covered)) == len(covered)


def search_more(kind: str, instance, routes, plans, oracle, limit, deadline):
    """Spend the remaining solver-execution or wall-clock budget.

    Basic local search exhausts its neighborhood before the larger neural
    budgets. Greedy + local search therefore restarts from a randomized
    restricted candidate list. Ruin-and-repair uses a new seed each chunk.
    The best complete solution is kept. The loop stops at the execution cap,
    at the deadline, or after two consecutive chunks that create no new
    solver execution.
    """
    if not routes:
        return routes, plans
    best_obj = sum(plan.duration_hours for plan in plans)
    best = ([tuple(route) for route in routes], list(plans))
    rng = random.Random(1)
    attempt = 0
    stale = 0
    while _room(oracle, limit, deadline) and attempt < 80:
        before = oracle.calls
        if kind == "greedy_ls" and attempt > 0:
            built, built_plans, left = rcl_construct(instance, oracle, limit, deadline, rng)
            if not left and built_plans:
                routes, plans = built, built_plans
        if not _room(oracle, limit, deadline):
            break
        chunk = 48 if limit is None else max(1, min(48, limit - oracle.calls))
        if kind == "greedy_ls":
            routes, plans, _imp, _diag = local_search(
                routes,
                plans,
                oracle,
                max_passes=2,
                evaluation_budget=chunk,
                operators="basic",
                deadline=deadline,
            )
        else:
            routes, plans, _imp, _diag = ruin_and_repair(
                routes,
                plans,
                oracle,
                instance,
                max_iterations=2,
                evaluation_budget=chunk,
                seed=1_000 + attempt,
            )
        if _complete(instance, routes) and plans:
            objective = sum(plan.duration_hours for plan in plans)
            if objective < best_obj:
                best_obj = objective
                best = ([tuple(route) for route in routes], list(plans))
                routes, plans = best
        attempt += 1
        if oracle.calls == before:
            stale += 1
            if stale < 2 or not _room(oracle, limit, deadline):
                if stale >= 2:
                    break
                continue
            # The neighborhood is cached. One randomized reconstruction can
            # create routes the cache has not seen. A second miss ends the search.
            built, built_plans, left = rcl_construct(instance, oracle, limit, deadline, rng)
            stale = 0
            if left or not built_plans or oracle.calls == before:
                break
            routes, plans = built, built_plans
        else:
            stale = 0
    return best[0], best[1]


def evaluate_instance(instance, kind: str, limit: int | None, seconds: float | None) -> dict:
    started = time.perf_counter()
    deadline = None if seconds is None else started + seconds
    oracle = FixedRouteOracle(instance)
    routes, plans, left = greedy_construct(instance, oracle, limit, deadline)
    feasible_construct = not left and bool(plans)
    if feasible_construct:
        routes, plans = search_more(kind, instance, routes, plans, oracle, limit, deadline)
        ok, reason = check_solution(instance, plans)
    else:
        ok, reason = False, f"{len(left)} customers unserved within budget"
    objective = sum(plan.duration_hours for plan in plans) if ok else float("inf")
    stats = oracle.stats()
    return {
        "feasible": int(ok),
        "objective_h": objective if ok else "",
        "objective_lit_h": literature_compatible_objective_hours(instance, objective) if ok else "",
        "runtime_s": time.perf_counter() - started,
        "oracle_calls": stats["oracle_solver_executions"],
        "oracle_requests": stats["oracle_requests"],
        "oracle_cache_hits": stats["oracle_cache_hits"],
        "reason": "" if ok else reason,
    }


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "executions"
    sizes = tuple(int(item) for item in (sys.argv[2:] or ["10", "20", "40", "80"]))
    budgets = neural_budgets()
    data_dir = ROOT / "dataset"
    done = set()
    rows = []
    if OUT.exists():
        with OUT.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        done = {
            (row["method"], row["budget_kind"], row["budget_protocol"], row["size"], row["instance"])
            for row in rows
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        paths = instances_of_size(data_dir, size)
        for protocol in ("P0", "P1", "P2"):
            budget = budgets[(size, protocol)]
            limit = int(round(budget["executions"])) if mode == "executions" else None
            seconds = budget["seconds"] if mode == "seconds" else None
            for kind in ("greedy_ls", "ruin_repair"):
                for path in paths:
                    instance = parse_instance(path)
                    key = (kind, mode, protocol, str(size), instance.name)
                    if key in done:
                        continue
                    result = evaluate_instance(instance, kind, limit, seconds)
                    row = {
                        "method": kind,
                        "budget_kind": mode,
                        "budget_protocol": protocol,
                        "budget_value": limit if limit is not None else round(seconds, 4),
                        "size": size,
                        "n_available": len(paths),
                        "instance": instance.name,
                        **result,
                    }
                    rows.append(row)
                    done.add(key)
                    with OUT.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    print(
                        {
                            "method": kind,
                            "size": size,
                            "protocol": protocol,
                            "instance": instance.name,
                            "feasible": result["feasible"],
                            "calls": result["oracle_calls"],
                        },
                        flush=True,
                    )


if __name__ == "__main__":
    main()
