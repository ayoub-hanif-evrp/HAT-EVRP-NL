"""Local-search operators with exact FRVCP evaluation and budget accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from charging import FixedRouteOracle, RoutePlan


@dataclass
class OperatorStats:
    attempted: int = 0
    feasible: int = 0
    provisional_updates: int = 0  # better than current pass incumbent
    accepted: int = 0  # applied moves (alias of improvements for operators)
    oracle_evals: int = 0
    improvement_h: float = 0.0


@dataclass
class SearchDiagnostics:
    evaluations: int = 0
    feasible_candidates: int = 0
    provisional_updates: int = 0
    improvements: int = 0  # accepted moves actually applied
    operators: dict[str, OperatorStats] = field(default_factory=dict)
    budget: int = 0
    schedule: str = "sequential"

    def ensure(self, name: str) -> OperatorStats:
        if name not in self.operators:
            self.operators[name] = OperatorStats()
        return self.operators[name]

    def as_dict(self) -> dict:
        return {
            "evaluations": self.evaluations,
            "feasible_candidates": self.feasible_candidates,
            "provisional_updates": self.provisional_updates,
            "improvements": self.improvements,
            "budget": self.budget,
            "schedule": self.schedule,
            "operators": {
                name: {
                    "attempted": stats.attempted,
                    "feasible": stats.feasible,
                    "provisional_updates": stats.provisional_updates,
                    "accepted": stats.accepted,
                    "oracle_evals": stats.oracle_evals,
                    "improvement_h": stats.improvement_h,
                }
                for name, stats in self.operators.items()
            },
        }


def local_search(
    routes: list[tuple[int, ...]],
    plans: list[RoutePlan],
    oracle: FixedRouteOracle,
    max_passes: int,
    evaluation_budget: int,
    operators: str = "basic",
) -> tuple[list[tuple[int, ...]], list[RoutePlan], int, SearchDiagnostics]:
    """Best-improvement neighborhoods with exact FRVCP evaluations.

    operators:
      - basic: 2-opt + relocate (Cycle-3)
      - strong: sequential 2-opt → relocate → swap → 2-opt* → eliminate
      - strong_balanced: round-robin operator schedule under the same budget
    """
    tolerance = 1e-10
    schedule = "balanced" if operators == "strong_balanced" else "sequential"
    use_strong = operators in {"strong", "strong_balanced"}
    diag = SearchDiagnostics(budget=evaluation_budget, schedule=schedule)

    def remaining() -> int:
        return evaluation_budget - diag.evaluations

    def solve(name: str, customer_route: tuple[int, ...]) -> RoutePlan | None:
        if remaining() <= 0:
            return None
        stats = diag.ensure(name)
        plan = oracle.solve(customer_route)
        diag.evaluations += 1
        stats.oracle_evals += 1
        return plan

    def consider(
        name: str,
        candidate_routes: list[tuple[int, ...]],
        candidate_plans: list[RoutePlan],
        best_objective: float,
        best_routes,
        best_plans,
        current_objective: float,
    ):
        stats = diag.ensure(name)
        stats.attempted += 1
        objective = sum(plan.duration_hours for plan in candidate_plans)
        stats.feasible += 1
        diag.feasible_candidates += 1
        if objective < best_objective - tolerance:
            stats.provisional_updates += 1
            diag.provisional_updates += 1
            stats.improvement_h += current_objective - objective
            return objective, candidate_routes, candidate_plans
        return best_objective, best_routes, best_plans

    def iter_two_opt(current_routes):
        for route_index, route in enumerate(current_routes):
            for begin in range(len(route) - 1):
                for end in range(begin + 1, len(route)):
                    yield ("two_opt", route_index, begin, end)

    def apply_two_opt(name, current_routes, current_plans, move, best_objective, best_routes, best_plans, current_objective):
        _, route_index, begin, end = move
        route = current_routes[route_index]
        candidate_route = route[:begin] + tuple(reversed(route[begin : end + 1])) + route[end + 1 :]
        if candidate_route == route:
            return best_objective, best_routes, best_plans
        plan = solve(name, candidate_route)
        if plan is None or not plan.feasible:
            return best_objective, best_routes, best_plans
        candidate_routes = current_routes.copy()
        candidate_plans = current_plans.copy()
        candidate_routes[route_index] = candidate_route
        candidate_plans[route_index] = plan
        return consider(
            name,
            candidate_routes,
            candidate_plans,
            best_objective,
            best_routes,
            best_plans,
            current_objective,
        )

    def iter_relocate(current_routes):
        for source_index, source in enumerate(current_routes):
            for customer_index, customer in enumerate(source):
                for target_index, target in enumerate(current_routes):
                    reduced = source[:customer_index] + source[customer_index + 1 :]
                    base_target = reduced if source_index == target_index else target
                    for position in range(len(base_target) + 1):
                        yield (
                            "relocate",
                            source_index,
                            customer_index,
                            customer,
                            target_index,
                            position,
                        )

    def apply_relocate(name, current_routes, current_plans, move, best_objective, best_routes, best_plans, current_objective):
        _, source_index, customer_index, customer, target_index, position = move
        source = current_routes[source_index]
        reduced = source[:customer_index] + source[customer_index + 1 :]
        target = current_routes[target_index]
        base_target = reduced if source_index == target_index else target
        inserted = base_target[:position] + (customer,) + base_target[position:]
        if source_index == target_index:
            if inserted == source:
                return best_objective, best_routes, best_plans
            if remaining() < 1:
                return best_objective, best_routes, best_plans
            plan = solve(name, inserted)
            if plan is None or not plan.feasible:
                return best_objective, best_routes, best_plans
            candidate_routes = current_routes.copy()
            candidate_plans = current_plans.copy()
            candidate_routes[source_index] = inserted
            candidate_plans[source_index] = plan
        else:
            need = 1 + (1 if reduced else 0)
            if remaining() < need:
                return best_objective, best_routes, best_plans
            inserted_plan = solve(name, inserted)
            if inserted_plan is None or not inserted_plan.feasible:
                return best_objective, best_routes, best_plans
            reduced_plan = None
            if reduced:
                reduced_plan = solve(name, reduced)
                if reduced_plan is None or not reduced_plan.feasible:
                    return best_objective, best_routes, best_plans
            candidate_routes = current_routes.copy()
            candidate_plans = current_plans.copy()
            candidate_routes[target_index] = inserted
            candidate_plans[target_index] = inserted_plan
            if reduced:
                candidate_routes[source_index] = reduced
                candidate_plans[source_index] = reduced_plan
            else:
                del candidate_routes[source_index]
                del candidate_plans[source_index]
        return consider(
            name,
            candidate_routes,
            candidate_plans,
            best_objective,
            best_routes,
            best_plans,
            current_objective,
        )

    def iter_swap(current_routes):
        for i in range(len(current_routes)):
            for j in range(i + 1, len(current_routes)):
                for ai, ca in enumerate(current_routes[i]):
                    for bj, cb in enumerate(current_routes[j]):
                        yield ("swap", i, j, ai, bj, ca, cb)

    def apply_swap(name, current_routes, current_plans, move, best_objective, best_routes, best_plans, current_objective):
        _, i, j, ai, bj, ca, cb = move
        if remaining() < 2:
            return best_objective, best_routes, best_plans
        ri = current_routes[i][:ai] + (cb,) + current_routes[i][ai + 1 :]
        rj = current_routes[j][:bj] + (ca,) + current_routes[j][bj + 1 :]
        pi = solve(name, ri)
        if pi is None or not pi.feasible:
            return best_objective, best_routes, best_plans
        pj = solve(name, rj)
        if pj is None or not pj.feasible:
            return best_objective, best_routes, best_plans
        candidate_routes = current_routes.copy()
        candidate_plans = current_plans.copy()
        candidate_routes[i], candidate_routes[j] = ri, rj
        candidate_plans[i], candidate_plans[j] = pi, pj
        return consider(
            name,
            candidate_routes,
            candidate_plans,
            best_objective,
            best_routes,
            best_plans,
            current_objective,
        )

    def iter_two_opt_star(current_routes):
        for i in range(len(current_routes)):
            for j in range(i + 1, len(current_routes)):
                for cut_i in range(len(current_routes[i]) + 1):
                    for cut_j in range(len(current_routes[j]) + 1):
                        for variant in (0, 1, 2):
                            yield ("two_opt_star", i, j, cut_i, cut_j, variant)

    def apply_two_opt_star(name, current_routes, current_plans, move, best_objective, best_routes, best_plans, current_objective):
        _, i, j, cut_i, cut_j, variant = move
        a_tail = current_routes[i][cut_i:]
        b_tail = current_routes[j][cut_j:]
        if variant == 1:
            a_tail = tuple(reversed(a_tail))
        elif variant == 2:
            b_tail = tuple(reversed(b_tail))
        ri = current_routes[i][:cut_i] + b_tail
        rj = current_routes[j][:cut_j] + a_tail
        if not ri or not rj or (ri == current_routes[i] and rj == current_routes[j]):
            return best_objective, best_routes, best_plans
        if remaining() < 2:
            return best_objective, best_routes, best_plans
        pi = solve(name, ri)
        if pi is None or not pi.feasible:
            return best_objective, best_routes, best_plans
        pj = solve(name, rj)
        if pj is None or not pj.feasible:
            return best_objective, best_routes, best_plans
        candidate_routes = current_routes.copy()
        candidate_plans = current_plans.copy()
        candidate_routes[i], candidate_routes[j] = ri, rj
        candidate_plans[i], candidate_plans[j] = pi, pj
        return consider(
            name,
            candidate_routes,
            candidate_plans,
            best_objective,
            best_routes,
            best_plans,
            current_objective,
        )

    def apply_eliminate(name, current_routes, current_plans, victim, best_objective, best_routes, best_plans, current_objective):
        if len(current_routes) < 2:
            return best_objective, best_routes, best_plans
        remaining_routes = [r for k, r in enumerate(current_routes) if k != victim]
        remaining_plans = [p for k, p in enumerate(current_plans) if k != victim]
        tentative_routes = [list(r) for r in remaining_routes]
        tentative_plans = list(remaining_plans)
        for customer in current_routes[victim]:
            best_ins = None
            best_delta = float("inf")
            best_t = -1
            best_plan = None
            for t, target in enumerate(tentative_routes):
                for pos in range(len(target) + 1):
                    if remaining() < 1:
                        return best_objective, best_routes, best_plans
                    inserted = tuple(target[:pos] + [customer] + target[pos:])
                    plan = solve(name, inserted)
                    if plan is None or not plan.feasible:
                        continue
                    delta = plan.duration_hours - tentative_plans[t].duration_hours
                    if delta < best_delta - tolerance:
                        best_delta = delta
                        best_ins = inserted
                        best_plan = plan
                        best_t = t
            if best_ins is None:
                return best_objective, best_routes, best_plans
            tentative_routes[best_t] = list(best_ins)
            tentative_plans[best_t] = best_plan
        return consider(
            name,
            [tuple(r) for r in tentative_routes],
            tentative_plans,
            best_objective,
            best_routes,
            best_plans,
            current_objective,
        )

    improvements = 0
    for _pass in range(max_passes):
        current_objective = sum(plan.duration_hours for plan in plans)
        best_objective = current_objective
        best_routes: list[tuple[int, ...]] | None = None
        best_plans: list[RoutePlan] | None = None

        if schedule == "sequential":
            names = ["two_opt", "relocate"]
            if use_strong:
                names.extend(["swap", "two_opt_star", "eliminate"])
            for name in names:
                if remaining() <= 0:
                    break
                if name == "two_opt":
                    for move in iter_two_opt(routes):
                        if remaining() <= 0:
                            break
                        best_objective, best_routes, best_plans = apply_two_opt(
                            name, routes, plans, move, best_objective, best_routes, best_plans, current_objective
                        )
                elif name == "relocate":
                    for move in iter_relocate(routes):
                        if remaining() <= 0:
                            break
                        best_objective, best_routes, best_plans = apply_relocate(
                            name, routes, plans, move, best_objective, best_routes, best_plans, current_objective
                        )
                elif name == "swap":
                    for move in iter_swap(routes):
                        if remaining() < 2:
                            break
                        best_objective, best_routes, best_plans = apply_swap(
                            name, routes, plans, move, best_objective, best_routes, best_plans, current_objective
                        )
                elif name == "two_opt_star":
                    for move in iter_two_opt_star(routes):
                        if remaining() < 2:
                            break
                        best_objective, best_routes, best_plans = apply_two_opt_star(
                            name, routes, plans, move, best_objective, best_routes, best_plans, current_objective
                        )
                elif name == "eliminate":
                    for victim in range(len(routes)):
                        if remaining() <= 0:
                            break
                        best_objective, best_routes, best_plans = apply_eliminate(
                            name, routes, plans, victim, best_objective, best_routes, best_plans, current_objective
                        )
        else:
            # Round-robin: one candidate batch slice per operator, cycling until budget ends.
            streams = {
                "two_opt": iter_two_opt(routes),
                "relocate": iter_relocate(routes),
            }
            if use_strong:
                streams["swap"] = iter_swap(routes)
                streams["two_opt_star"] = iter_two_opt_star(routes)
                streams["eliminate"] = iter(range(len(routes)))
            alive = set(streams)
            order = list(streams.keys())
            while remaining() > 0 and alive:
                progressed = False
                for name in order:
                    if name not in alive or remaining() <= 0:
                        continue
                    try:
                        move = next(streams[name])
                    except StopIteration:
                        alive.discard(name)
                        continue
                    progressed = True
                    if name == "two_opt":
                        best_objective, best_routes, best_plans = apply_two_opt(
                            name, routes, plans, move, best_objective, best_routes, best_plans, current_objective
                        )
                    elif name == "relocate":
                        best_objective, best_routes, best_plans = apply_relocate(
                            name, routes, plans, move, best_objective, best_routes, best_plans, current_objective
                        )
                    elif name == "swap":
                        best_objective, best_routes, best_plans = apply_swap(
                            name, routes, plans, move, best_objective, best_routes, best_plans, current_objective
                        )
                    elif name == "two_opt_star":
                        best_objective, best_routes, best_plans = apply_two_opt_star(
                            name, routes, plans, move, best_objective, best_routes, best_plans, current_objective
                        )
                    elif name == "eliminate":
                        best_objective, best_routes, best_plans = apply_eliminate(
                            name, routes, plans, move, best_objective, best_routes, best_plans, current_objective
                        )
                if not progressed:
                    break

        if best_routes is None or best_plans is None:
            break
        routes, plans = best_routes, best_plans
        improvements += 1
        diag.improvements += 1
        # Credit the operator that produced the applied incumbent if unique.
        for stats in diag.operators.values():
            if stats.provisional_updates > stats.accepted:
                stats.accepted += 1
                break
        # Best-improvement: restart pass after one accepted move.
    return routes, plans, improvements, diag


def _route_centroid(instance, route: tuple[int, ...]) -> tuple[float, float]:
    if not route:
        return instance.nodes[instance.depot].x, instance.nodes[instance.depot].y
    xs = [instance.nodes[c].x for c in route]
    ys = [instance.nodes[c].y for c in route]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def ruin_and_repair(
    routes: list[tuple[int, ...]],
    plans: list[RoutePlan],
    oracle: FixedRouteOracle,
    instance,
    max_iterations: int = 8,
    evaluation_budget: int = 400,
    seed: int = 0,
    destroy_min: int = 2,
    destroy_max: int = 4,
) -> tuple[list[tuple[int, ...]], list[RoutePlan], int, SearchDiagnostics]:
    """Bounded ruin-and-repair with charging-aware cheapest/regret repair.

    Destroy operators (sampled): random customers, geographic cluster,
    longest-route customers, or remove a short complete route.
    Repair: cheapest feasible insertion then regret-2 fallback.
    """
    rng = random.Random(seed)
    diag = SearchDiagnostics(budget=evaluation_budget, schedule="ruin_repair")
    tolerance = 1e-10
    best_routes = [tuple(r) for r in routes]
    best_plans = list(plans)
    best_obj = sum(p.duration_hours for p in best_plans)
    improvements = 0

    def remaining() -> int:
        return evaluation_budget - diag.evaluations

    def solve(name: str, customer_route: tuple[int, ...]) -> RoutePlan | None:
        if remaining() <= 0:
            return None
        stats = diag.ensure(name)
        plan = oracle.solve(customer_route)
        diag.evaluations += 1
        stats.oracle_evals += 1
        return plan

    def destroy(current_routes: list[tuple[int, ...]]) -> tuple[list[tuple[int, ...]], list[int]]:
        customers = [c for r in current_routes for c in r]
        if len(customers) < 2:
            return current_routes, []
        mode = rng.choice(("random", "geo", "long", "short_route"))
        remove: set[int] = set()
        k = min(len(customers), rng.randint(destroy_min, destroy_max))
        if mode == "short_route" and len(current_routes) >= 2:
            lengths = [(len(r), i) for i, r in enumerate(current_routes)]
            lengths.sort()
            victim = lengths[0][1]
            remove = set(current_routes[victim])
        elif mode == "long":
            durations = []
            for i, r in enumerate(current_routes):
                # Prefer longer customer sequences as proxy for inefficiency.
                durations.append((len(r), i))
            durations.sort(reverse=True)
            long_idx = durations[0][1]
            pool = list(current_routes[long_idx])
            rng.shuffle(pool)
            remove = set(pool[:k])
        elif mode == "geo":
            seed_c = rng.choice(customers)
            sx, sy = instance.nodes[seed_c].x, instance.nodes[seed_c].y
            ranked = sorted(
                customers,
                key=lambda c: math.hypot(
                    instance.nodes[c].x - sx, instance.nodes[c].y - sy
                ),
            )
            remove = set(ranked[:k])
        else:
            remove = set(rng.sample(customers, k))

        ruined = []
        for route in current_routes:
            kept = tuple(c for c in route if c not in remove)
            if kept:
                ruined.append(kept)
        return ruined, sorted(remove)

    def cheapest_insert(
        name: str,
        base_routes: list[tuple[int, ...]],
        base_plans: list[RoutePlan],
        customer: int,
    ) -> tuple[list[tuple[int, ...]], list[RoutePlan], float] | None:
        best = None
        # Try existing routes.
        for t, target in enumerate(base_routes):
            for pos in range(len(target) + 1):
                if remaining() < 1:
                    return best
                inserted = target[:pos] + (customer,) + target[pos:]
                plan = solve(name, inserted)
                if plan is None or not plan.feasible:
                    continue
                diag.feasible_candidates += 1
                diag.ensure(name).feasible += 1
                obj = (
                    sum(p.duration_hours for i, p in enumerate(base_plans) if i != t)
                    + plan.duration_hours
                )
                cand_routes = base_routes.copy()
                cand_plans = base_plans.copy()
                cand_routes[t] = inserted
                cand_plans[t] = plan
                if best is None or obj < best[2] - tolerance:
                    best = (cand_routes, cand_plans, obj)
        # New route.
        if remaining() >= 1:
            plan = solve(name, (customer,))
            if plan is not None and plan.feasible:
                diag.feasible_candidates += 1
                diag.ensure(name).feasible += 1
                cand_routes = base_routes + [(customer,)]
                cand_plans = base_plans + [plan]
                obj = sum(p.duration_hours for p in cand_plans)
                if best is None or obj < best[2] - tolerance:
                    best = (cand_routes, cand_plans, obj)
        return best

    def regret2_order(removed: list[int], base_routes, base_plans) -> list[int]:
        """Order removed customers by insertion regret (approx, budget-light)."""
        if len(removed) <= 1 or remaining() < 4:
            return removed
        scored = []
        for customer in removed:
            costs = []
            for t, target in enumerate(base_routes):
                for pos in (0, len(target) // 2, len(target)):
                    if remaining() < 1:
                        break
                    inserted = target[:pos] + (customer,) + target[pos:]
                    plan = solve("repair_regret", inserted)
                    if plan is not None and plan.feasible:
                        delta = plan.duration_hours - base_plans[t].duration_hours
                        costs.append(delta)
            if remaining() >= 1:
                plan = solve("repair_regret", (customer,))
                if plan is not None and plan.feasible:
                    costs.append(plan.duration_hours)
            costs.sort()
            if len(costs) >= 2:
                regret = costs[1] - costs[0]
            elif costs:
                regret = costs[0]
            else:
                regret = float("inf")
            scored.append((regret, customer))
        scored.sort(reverse=True)
        return [c for _, c in scored]

    current_routes = [tuple(r) for r in best_routes]
    current_plans = list(best_plans)
    for _it in range(max_iterations):
        if remaining() <= 0:
            break
        ruined, removed = destroy(current_routes)
        if not removed:
            continue
        # Re-optimize remaining routes after destroy.
        rebuilt_routes: list[tuple[int, ...]] = []
        rebuilt_plans: list[RoutePlan] = []
        feasible_base = True
        for route in ruined:
            plan = solve("ruin", route)
            if plan is None or not plan.feasible:
                feasible_base = False
                break
            rebuilt_routes.append(route)
            rebuilt_plans.append(plan)
        if not feasible_base:
            continue
        order = regret2_order(removed, rebuilt_routes, rebuilt_plans)
        repaired_routes = rebuilt_routes
        repaired_plans = rebuilt_plans
        ok = True
        for customer in order:
            diag.ensure("repair").attempted += 1
            result = cheapest_insert(
                "repair", repaired_routes, repaired_plans, customer
            )
            if result is None:
                ok = False
                break
            repaired_routes, repaired_plans, _ = result
        if not ok:
            continue
        # Exact coverage check.
        covered = sorted(c for r in repaired_routes for c in r)
        if covered != sorted(instance.customers):
            continue
        obj = sum(p.duration_hours for p in repaired_plans)
        diag.ensure("repair").feasible += 1
        if obj < best_obj - tolerance:
            diag.provisional_updates += 1
            diag.ensure("repair").provisional_updates += 1
            best_obj = obj
            best_routes = repaired_routes
            best_plans = repaired_plans
            current_routes = repaired_routes
            current_plans = repaired_plans
            improvements += 1
            diag.improvements += 1
            diag.ensure("repair").accepted += 1
            diag.ensure("repair").improvement_h += (
                sum(p.duration_hours for p in plans) - obj
            )
        else:
            # Keep exploring from current (not only best) occasionally.
            if rng.random() < 0.25:
                current_routes = repaired_routes
                current_plans = repaired_plans
    return best_routes, best_plans, improvements, diag


def assemble_route_pool(
    candidate_solutions: list[tuple[list[tuple[int, ...]], list[RoutePlan]]],
    oracle: FixedRouteOracle,
    instance,
    evaluation_budget: int = 200,
) -> tuple[list[tuple[int, ...]], list[RoutePlan], int, SearchDiagnostics]:
    """Greedy set-cover style assembly from unique feasible customer routes.

    Method-specific pools should be labeled separately from portfolios.
    """
    diag = SearchDiagnostics(budget=evaluation_budget, schedule="route_pool")
    # Best verified plan per customer sequence.
    pool: dict[tuple[int, ...], RoutePlan] = {}
    for routes, plans in candidate_solutions:
        for route, plan in zip(routes, plans):
            if not plan.feasible:
                continue
            key = tuple(route)
            prev = pool.get(key)
            if prev is None or plan.duration_hours < prev.duration_hours - 1e-12:
                pool[key] = plan
    if not pool:
        return [], [], 0, diag

    remaining_customers = set(instance.customers)
    chosen_routes: list[tuple[int, ...]] = []
    chosen_plans: list[RoutePlan] = []
    improvements = 0
    while remaining_customers:
        best_key = None
        best_score = float("inf")
        for key, plan in pool.items():
            if any(c not in remaining_customers for c in key):
                continue
            if not key:
                continue
            # Prefer routes covering more remaining customers per hour.
            score = plan.duration_hours / len(key)
            if score < best_score:
                best_score = score
                best_key = key
        if best_key is None:
            # Force singleton insertion for leftover customers.
            customer = min(remaining_customers)
            if diag.evaluations >= evaluation_budget:
                break
            plan = oracle.solve((customer,))
            diag.evaluations += 1
            if plan is None or not plan.feasible:
                break
            best_key = (customer,)
            pool[best_key] = plan
        chosen_routes.append(best_key)
        chosen_plans.append(pool[best_key])
        remaining_customers -= set(best_key)
        improvements += 1
    if remaining_customers:
        return [], [], 0, diag
    return chosen_routes, chosen_plans, improvements, diag
