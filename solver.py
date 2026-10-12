"""Compact randomized-greedy baseline for customer assignment and ordering."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

from charging import FixedRouteOracle, RoutePlan, check_route, check_solution
from data import Instance


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


def local_search(
    routes: list[tuple[int, ...]],
    plans: list[RoutePlan],
    oracle: FixedRouteOracle,
    max_passes: int,
    evaluation_budget: int,
) -> tuple[list[tuple[int, ...]], list[RoutePlan], int]:
    """Best-improvement 2-opt and relocate using exact charging evaluations."""
    improvements = 0
    evaluations = 0
    tolerance = 1e-10

    for _ in range(max_passes):
        current_objective = sum(plan.duration_hours for plan in plans)
        best_objective = current_objective
        best_routes: list[tuple[int, ...]] | None = None
        best_plans: list[RoutePlan] | None = None

        # Intra-route 2-opt.
        for route_index, route in enumerate(routes):
            for begin in range(len(route) - 1):
                for end in range(begin + 1, len(route)):
                    if evaluations >= evaluation_budget:
                        break
                    candidate_route = (
                        route[:begin] + tuple(reversed(route[begin : end + 1])) + route[end + 1 :]
                    )
                    if candidate_route == route:
                        continue
                    candidate_plan = oracle.solve(candidate_route)
                    evaluations += 1
                    if not candidate_plan.feasible:
                        continue
                    objective = (
                        current_objective
                        - plans[route_index].duration_hours
                        + candidate_plan.duration_hours
                    )
                    if objective < best_objective - tolerance:
                        candidate_routes = routes.copy()
                        candidate_plans = plans.copy()
                        candidate_routes[route_index] = candidate_route
                        candidate_plans[route_index] = candidate_plan
                        best_objective = objective
                        best_routes, best_plans = candidate_routes, candidate_plans
                if evaluations >= evaluation_budget:
                    break
            if evaluations >= evaluation_budget:
                break

        # Relocate one customer within or between routes; empty routes disappear.
        if evaluations < evaluation_budget:
            for source_index, source in enumerate(routes):
                for customer_index, customer in enumerate(source):
                    reduced = source[:customer_index] + source[customer_index + 1 :]
                    for target_index, target in enumerate(routes):
                        base_target = reduced if source_index == target_index else target
                        for position in range(len(base_target) + 1):
                            if evaluations >= evaluation_budget:
                                break
                            inserted = (
                                base_target[:position] + (customer,) + base_target[position:]
                            )
                            if source_index == target_index:
                                if inserted == source:
                                    continue
                                candidate_plan = oracle.solve(inserted)
                                evaluations += 1
                                if not candidate_plan.feasible:
                                    continue
                                candidate_routes = routes.copy()
                                candidate_plans = plans.copy()
                                candidate_routes[source_index] = inserted
                                candidate_plans[source_index] = candidate_plan
                            else:
                                inserted_plan = oracle.solve(inserted)
                                evaluations += 1
                                if not inserted_plan.feasible:
                                    continue
                                reduced_plan = oracle.solve(reduced) if reduced else None
                                evaluations += int(bool(reduced))
                                if reduced and not reduced_plan.feasible:
                                    continue
                                candidate_routes = routes.copy()
                                candidate_plans = plans.copy()
                                candidate_routes[target_index] = inserted
                                candidate_plans[target_index] = inserted_plan
                                if reduced:
                                    candidate_routes[source_index] = reduced
                                    candidate_plans[source_index] = reduced_plan
                                else:
                                    del candidate_routes[source_index]
                                    del candidate_plans[source_index]
                            objective = sum(
                                candidate.duration_hours for candidate in candidate_plans
                            )
                            if objective < best_objective - tolerance:
                                best_objective = objective
                                best_routes, best_plans = candidate_routes, candidate_plans
                        if evaluations >= evaluation_budget:
                            break
                    if evaluations >= evaluation_budget:
                        break
                if evaluations >= evaluation_budget:
                    break

        if best_routes is None or best_plans is None:
            break
        routes, plans = best_routes, best_plans
        improvements += 1
    return routes, plans, improvements


def randomized_greedy(
    instance: Instance,
    restarts: int = 5,
    seed: int = 42,
    rcl_size: int = 3,
    local_search_passes: int = 2,
    local_search_budget: int = 2000,
    deterministic_first: bool = True,
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
    routes, plans, improvements = local_search(
        routes,
        plans,
        oracle,
        max_passes=local_search_passes,
        evaluation_budget=local_search_budget,
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
