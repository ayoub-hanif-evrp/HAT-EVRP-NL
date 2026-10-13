"""Compact randomized-greedy baseline for customer assignment and ordering."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

from charging import FixedRouteOracle, RoutePlan, check_route, check_solution
from data import Instance
from solver_ls import (
    SearchDiagnostics,
    assemble_route_pool,
    local_search,
    ruin_and_repair,
)


@dataclass(frozen=True)
class BaselineResult:
    feasible: bool
    objective_hours: float
    customer_routes: tuple[tuple[int, ...], ...]
    plans: tuple[RoutePlan, ...]
    routes: int
    charging_stops: int
    charging_hours: float
    oracle_calls: int
    runtime_seconds: float
    local_search_seconds: float
    local_improvements: int
    reason: str = ""
    unique_constructions: int = 0


def randomized_greedy(
    instance: Instance,
    restarts: int = 5,
    seed: int = 42,
    rcl_size: int = 3,
    local_search_passes: int = 2,
    local_search_budget: int = 2000,
    deterministic_first: bool = True,
    local_operators: str = "basic",
) -> BaselineResult:
    """Append customers by least feasible duration increase, with an RCL.

    When ``deterministic_first`` is True, only the first restart of *this call*
    is pure greedy. All other restarts sample an RCL with an independent RNG
    seeded by ``seed + restart``. Wall-clock single-trial loops must pass
    ``deterministic_first=False`` (or call once with True, then False) so that
    changing the seed actually changes constructions.
    """
    start = time.perf_counter()
    oracle = FixedRouteOracle(instance)
    best: tuple[float, list[tuple[int, ...]], list[RoutePlan]] | None = None
    failure_reason = ""
    unique_constructions: set[tuple[tuple[int, ...], ...]] = set()

    for restart in range(restarts):
        rng = random.Random(seed + restart)
        unassigned = set(instance.customers)
        routes: list[tuple[int, ...]] = []
        plans: list[RoutePlan] = []

        while unassigned:
            sequence: tuple[int, ...] = ()
            current_duration = 0.0
            final_plan: RoutePlan | None = None

            while unassigned:
                options: list[tuple[float, int, RoutePlan]] = []
                for customer in sorted(unassigned):
                    plan = oracle.solve((*sequence, customer))
                    if plan.feasible:
                        options.append(
                            (plan.duration_hours - current_duration, customer, plan)
                        )
                if not options:
                    break
                options.sort(key=lambda option: (option[0], option[1]))
                use_deterministic = deterministic_first and restart == 0
                if use_deterministic:
                    selected = options[0]
                else:
                    selected = rng.choice(options[: min(rcl_size, len(options))])
                _, customer, final_plan = selected
                sequence = (*sequence, customer)
                current_duration = final_plan.duration_hours
                unassigned.remove(customer)

            if not sequence or final_plan is None:
                failure_reason = (
                    f"no feasible singleton route for customers {sorted(unassigned)}"
                )
                break
            routes.append(sequence)
            plans.append(final_plan)

        if unassigned:
            continue
        valid, reason = check_solution(instance, plans)
        if not valid:
            failure_reason = reason
            continue
        unique_constructions.add(tuple(routes))
        objective = sum(plan.duration_hours for plan in plans)
        if best is None or objective < best[0]:
            best = objective, routes, plans

    elapsed = time.perf_counter() - start
    if best is None:
        return BaselineResult(
            False,
            math.inf,
            (),
            (),
            0,
            0,
            0.0,
            oracle.calls,
            elapsed,
            0.0,
            0,
            failure_reason or "all restarts failed",
            len(unique_constructions),
        )

    objective, routes, plans = best
    local_start = time.perf_counter()
    routes, plans, improvements, _diag = local_search(
        routes,
        plans,
        oracle,
        max_passes=local_search_passes,
        evaluation_budget=local_search_budget,
        operators=local_operators,
    )
    local_seconds = time.perf_counter() - local_start
    valid, reason = check_solution(instance, plans)
    if not valid:
        return BaselineResult(
            False,
            math.inf,
            (),
            (),
            0,
            0,
            0.0,
            oracle.calls,
            time.perf_counter() - start,
            local_seconds,
            improvements,
            f"local-search result invalid: {reason}",
            len(unique_constructions),
        )
    objective = sum(plan.duration_hours for plan in plans)
    charging_stops = 0
    charging_hours = 0.0
    for plan in plans:
        check = check_route(instance, plan.stops)
        charging_stops += check.charging_stops
        charging_hours += check.charging_hours
    return BaselineResult(
        True,
        objective,
        tuple(routes),
        tuple(plans),
        len(routes),
        charging_stops,
        charging_hours,
        oracle.calls,
        time.perf_counter() - start,
        local_seconds,
        improvements,
        "",
        len(unique_constructions),
    )
