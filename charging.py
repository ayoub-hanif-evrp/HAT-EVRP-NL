"""Exact benchmark charging interpolation, fixed-route oracle, and checker."""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import math

from data import ChargingCurve, Instance


ConcreteRoute = tuple[tuple[int, float | None], ...]


def total_customer_service_hours(instance: Instance) -> float:
    """Constant total service time over all customers (Montoya instances: 0.5 h each)."""
    return float(sum(instance.service_times[customer] for customer in instance.customers))


def literature_compatible_objective_hours(
    instance: Instance, full_duration_hours: float
) -> float:
    """Travel + charging duration for literature BKS comparison.

    Montoya et al. (2017) minimize travel + charging time in the objective, while
    including service time only in the route-duration constraint. Our simulator
    reports full duration (travel + service + charging). Subtracting the constant
    total customer service time yields the literature-compatible objective when
    every customer is served exactly once.
    """
    return float(full_duration_hours) - total_customer_service_hours(instance)


@dataclass(frozen=True)
class RoutePlan:
    feasible: bool
    duration_hours: float
    stops: ConcreteRoute | None
    reason: str = ""


@dataclass(frozen=True)
class RouteCheck:
    feasible: bool
    duration_hours: float
    charging_hours: float
    end_soc_wh: float
    customers: tuple[int, ...]
    charging_stops: int
    reason: str = ""


def cumulative_charging_time(curve: ChargingCurve, energy_wh: float) -> float:
    """C_s(q): empty-to-energy charging time by piecewise-linear interpolation."""
    tolerance = 1e-8
    if energy_wh < -tolerance or energy_wh > curve.energies[-1] + tolerance:
        raise ValueError(f"Energy {energy_wh} outside charging curve")
    energy_wh = min(max(energy_wh, curve.energies[0]), curve.energies[-1])
    if energy_wh == curve.energies[-1]:
        return curve.times[-1]
    segment = bisect_right(curve.energies, energy_wh) - 1
    e0, e1 = curve.energies[segment], curve.energies[segment + 1]
    t0, t1 = curve.times[segment], curve.times[segment + 1]
    return t0 + (energy_wh - e0) * (t1 - t0) / (e1 - e0)


def charging_time(curve: ChargingCurve, q_in: float, q_out: float) -> float:
    if q_out + 1e-8 < q_in:
        raise ValueError("Charging cannot decrease battery energy")
    return cumulative_charging_time(curve, q_out) - cumulative_charging_time(
        curve, q_in
    )


def to_frvcpy(instance: Instance) -> dict:
    """Build solver input with an inaccessible helper for the terminal depot.

    frvcpy requires every fixed-route node (including the destination depot) to
    have a reachable CS after it. Its stock translator makes the depot a normal
    CS, which also permits mid-route depot charging. The helper below is a depot
    copy used only for lower bounds; FixedRouteOracle removes it from the
    insertable-CS list before solving.
    """
    helper = len(instance.nodes)
    time_matrix = [
        [*row, instance.travel_hours[node][instance.depot]]
        for node, row in enumerate(instance.travel_hours)
    ]
    time_matrix.append([*instance.travel_hours[instance.depot], 0.0])
    energy_matrix = [
        [*row, instance.energy_wh[node][instance.depot]]
        for node, row in enumerate(instance.energy_wh)
    ]
    energy_matrix.append([*instance.energy_wh[instance.depot], 0.0])
    fastest_type = max(
        instance.curves,
        key=lambda station_type: (
            instance.curves[station_type].energies[1]
            - instance.curves[station_type].energies[0]
        )
        / (
            instance.curves[station_type].times[1]
            - instance.curves[station_type].times[0]
        ),
    )
    return {
        "max_q": instance.battery_wh,
        "t_max": instance.max_route_hours,
        "css": [
            {
                "node_id": station,
                "cs_type": instance.nodes[station].station_type,
            }
            for station in instance.stations
        ]
        + [{"node_id": helper, "cs_type": fastest_type}],
        "process_times": [*instance.service_times, 0.0],
        "breakpoints_by_type": [
            {
                "cs_type": station_type,
                "time": list(curve.times),
                "charge": list(curve.energies),
            }
            for station_type, curve in instance.curves.items()
        ],
        "energy_matrix": energy_matrix,
        "time_matrix": time_matrix,
    }


class FixedRouteOracle:
    """Cached wrapper around frvcpy's Froger-labeling charging optimizer."""

    def __init__(self, instance: Instance):
        try:
            from frvcpy.core import NodeType
            from frvcpy.solver import Solver
        except ImportError as exc:
            raise RuntimeError("Phase 1 requires the installed frvcpy package") from exc
        self._solver_class = Solver
        self._customer_node_type = NodeType.CUSTOMER
        self.instance = instance
        self._oracle_instance = to_frvcpy(instance)
        self._terminal_helper = len(instance.nodes)
        self._cache: dict[tuple[int, ...], RoutePlan] = {}
        self.calls = 0  # actual frvcpy solver executions (cache misses)
        self.requests = 0  # logical solve requests including cache hits
        self.cache_hits = 0

    @property
    def solver_executions(self) -> int:
        """Alias: true charging-optimizer runs (not cache hits)."""
        return self.calls

    def stats(self) -> dict:
        return {
            "oracle_requests": self.requests,
            "oracle_cache_hits": self.cache_hits,
            "oracle_solver_executions": self.calls,
        }

    def solve(self, customers: tuple[int, ...] | list[int]) -> RoutePlan:
        customer_sequence = tuple(customers)
        self.requests += 1
        if customer_sequence in self._cache:
            self.cache_hits += 1
            return self._cache[customer_sequence]
        if (
            len(set(customer_sequence)) != len(customer_sequence)
            or any(node not in self.instance.customers for node in customer_sequence)
        ):
            plan = RoutePlan(False, math.inf, None, "invalid customer sequence")
            self._cache[customer_sequence] = plan
            return plan

        route = [self.instance.depot, *customer_sequence, self.instance.depot]
        self.calls += 1
        frvcp_solver = self._solver_class(
            self._oracle_instance,
            route,
            self.instance.battery_wh,
            multi_insert=True,
        )
        # Retain the helper in cs_id_to_type for get_min_energy_to_cs(), but
        # prevent it from becoming an actual charging stop in graph G'.
        frvcp_solver.instance.cs_details = [
            cs
            for cs in frvcp_solver.instance.cs_details
            if cs["node_id"] != self._terminal_helper
        ]
        frvcp_solver.instance.n_cs = len(frvcp_solver.instance.cs_details)
        frvcp_solver.instance.nodes_g[
            self._terminal_helper
        ].type = self._customer_node_type
        duration, stops = frvcp_solver.solve()
        if stops is None or not math.isfinite(duration):
            plan = RoutePlan(False, math.inf, None, "frvcpy found no feasible plan")
        else:
            concrete = tuple(
                (int(node), None if amount is None else float(amount))
                for node, amount in stops
            )
            if any(node == self._terminal_helper for node, _ in concrete):
                plan = RoutePlan(False, math.inf, concrete, "terminal helper was inserted")
                self._cache[customer_sequence] = plan
                return plan
            check = check_route(self.instance, concrete, customer_sequence)
            objective_error = abs(check.duration_hours - float(duration))
            if not check.feasible:
                plan = RoutePlan(False, math.inf, concrete, check.reason)
            elif objective_error > 1e-7:
                plan = RoutePlan(
                    False,
                    math.inf,
                    concrete,
                    f"oracle/checker duration mismatch {objective_error:.3g} h",
                )
            else:
                plan = RoutePlan(True, check.duration_hours, concrete)
        self._cache[customer_sequence] = plan
        return plan


def check_route(
    instance: Instance,
    stops: ConcreteRoute,
    expected_customers: tuple[int, ...] | None = None,
    tolerance: float = 1e-7,
) -> RouteCheck:
    """Independently forward-simulate one concrete route."""

    def fail(reason: str) -> RouteCheck:
        return RouteCheck(False, duration, charge_duration, soc, tuple(seen), n_charges, reason)

    duration = 0.0
    charge_duration = 0.0
    soc = instance.battery_wh
    seen: list[int] = []
    n_charges = 0
    if len(stops) < 2:
        return fail("route has fewer than two stops")
    if stops[0][0] != instance.depot or stops[-1][0] != instance.depot:
        return fail("route does not start and end at depot")

    for index, (node, amount) in enumerate(stops):
        if node < 0 or node >= len(instance.nodes):
            return fail(f"unknown node {node}")
        kind = instance.nodes[node].kind
        # Depot may only appear at the endpoints (start/end).
        if kind == 0 and index not in (0, len(stops) - 1):
            return fail(f"interior depot visit at stop {index}")
        if kind == 1:
            seen.append(node)
            duration += instance.service_times[node]

        if amount is not None:
            if kind != 2:
                return fail(f"charging specified at non-station {node}")
            if amount < -tolerance:
                return fail(f"negative charge at station {node}")
            q_out = soc + max(amount, 0.0)
            if q_out > instance.battery_wh + tolerance:
                return fail(f"battery exceeds capacity at station {node}")
            curve = instance.curves[instance.nodes[node].station_type]
            added_time = charging_time(curve, soc, min(q_out, instance.battery_wh))
            duration += added_time
            charge_duration += added_time
            soc = min(q_out, instance.battery_wh)
            if amount > tolerance:
                n_charges += 1

        if index + 1 < len(stops):
            next_node = stops[index + 1][0]
            required = instance.energy_wh[node][next_node]
            if required > soc + tolerance:
                return fail(f"negative SOC on arc {node}->{next_node}")
            soc = max(0.0, soc - required)
            duration += instance.travel_hours[node][next_node]

    expected = tuple(expected_customers) if expected_customers is not None else None
    if expected is not None and tuple(seen) != expected:
        return fail(f"customer order {tuple(seen)} != expected {expected}")
    if len(seen) != len(set(seen)):
        return fail("customer repeated within route")
    # Empty depot→depot with no customers is not a valid service route.
    if not seen:
        return fail("empty depot-to-depot route")
    if duration > instance.max_route_hours + tolerance:
        return fail(
            f"route duration {duration:.9f} exceeds {instance.max_route_hours:.9f}"
        )
    return RouteCheck(
        True, duration, charge_duration, soc, tuple(seen), n_charges
    )


def check_solution(instance: Instance, plans: list[RoutePlan]) -> tuple[bool, str]:
    all_customers: list[int] = []
    for route_number, plan in enumerate(plans):
        if not plan.feasible or plan.stops is None:
            return False, f"route {route_number} has no feasible plan"
        check = check_route(instance, plan.stops)
        if not check.feasible:
            return False, f"route {route_number}: {check.reason}"
        all_customers.extend(check.customers)
    if len(all_customers) != len(set(all_customers)):
        return False, "a customer appears on multiple routes"
    if set(all_customers) != set(instance.customers):
        missing = sorted(set(instance.customers) - set(all_customers))
        extra = sorted(set(all_customers) - set(instance.customers))
        return False, f"coverage mismatch; missing={missing}, extra={extra}"
    return True, ""
