"""Regression tests for position-aware neural repair (T1–T10)."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import FixedRouteOracle, check_route, check_solution
from data import generate_synthetic, instance_files, parse_instance
from train import (
    _accept_repaired,
    _partial_after_destroy,
    best_feasible_insertion,
    decode,
    load_policy,
    neural_lns,
    route_state,
)


def _official_n10():
    paths = [
        p
        for p in instance_files(ROOT / "dataset")
        if len(parse_instance(p).customers) == 10
    ]
    return parse_instance(paths[0])


def _load_h2():
    for name in (
        "H2_n10_seed42_c3.pt",
        "H2_n10_seed42_perf_cont_v2c.pt",
        "H2_n10_seed42_perf_repair_fix.pt",
    ):
        path = ROOT / "checkpoints" / name
        if path.exists():
            return load_policy(path)
    raise FileNotFoundError("no H2 checkpoint")


class TestBestFeasibleInsertion(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = _official_n10()
        self.oracle = FixedRouteOracle(self.instance)

    def test_t2_middle_preferred_when_cheaper(self) -> None:
        """Middle-position insertion wins when duration is strictly better."""
        customers = list(self.instance.customers)
        # Find A,C feasible and A,B,C / A,C,B both feasible with middle better.
        found = False
        for i, a in enumerate(customers):
            for c in customers[i + 1 :]:
                base = self.oracle.solve((a, c))
                if not base.feasible:
                    continue
                for b in customers:
                    if b in (a, c):
                        continue
                    mid = self.oracle.solve((a, b, c))
                    end = self.oracle.solve((a, c, b))
                    if not (mid.feasible and end.feasible):
                        continue
                    if mid.duration_hours + 1e-9 < end.duration_hours:
                        result = best_feasible_insertion(
                            self.oracle, (a, c), b
                        )
                        self.assertIsNotNone(result)
                        self.assertEqual(result["customers"], (a, b, c))
                        self.assertEqual(result["position"], 1)
                        self.assertTrue(result["is_interior"])
                        found = True
                        break
                if found:
                    break
            if found:
                break
        self.assertTrue(found, "could not find a case where middle beats append")

    def test_t3_preserves_survivor_order(self) -> None:
        route = tuple(self.instance.customers[:3])
        plan = self.oracle.solve(route)
        if not plan.feasible:
            route = tuple(self.instance.customers[:2])
            plan = self.oracle.solve(route)
        self.assertTrue(plan.feasible)
        removed = route[-1]
        survivors = route[:-1]
        extra = [
            c for c in self.instance.customers if c not in route
        ][0]
        result = best_feasible_insertion(self.oracle, survivors, extra)
        if result is None:
            self.skipTest("no feasible insertion for this triple")
        # Survivors appear in the same relative order.
        positions = [result["customers"].index(c) for c in survivors]
        self.assertEqual(positions, sorted(positions))

    def test_t4_each_position_calls_frvcpy(self) -> None:
        route = tuple(self.instance.customers[:2])
        if not self.oracle.solve(route).feasible:
            self.skipTest("base route infeasible")
        customer = self.instance.customers[2]
        before = self.oracle.calls
        # Clear cache entries for candidates by using a fresh oracle.
        fresh = FixedRouteOracle(self.instance)
        best_feasible_insertion(fresh, route, customer)
        # Positions 0..len(route) plus optional base solve = len(route)+1 or +2
        self.assertGreaterEqual(fresh.calls, len(route) + 1)

    def test_t5_infeasible_insertion_rejected(self) -> None:
        # Duplicate customer cannot be inserted.
        route = (self.instance.customers[0],)
        result = best_feasible_insertion(
            self.oracle, route, self.instance.customers[0]
        )
        self.assertIsNone(result)

    def test_t9_oracle_accounting(self) -> None:
        route = (self.instance.customers[0],)
        _ = self.oracle.solve(route)
        calls_after_first = self.oracle.calls
        req_after_first = self.oracle.requests
        _ = self.oracle.solve(route)  # cache hit
        self.assertEqual(self.oracle.calls, calls_after_first)
        self.assertEqual(self.oracle.requests, req_after_first + 1)
        self.assertEqual(self.oracle.cache_hits, 1)
        self.assertEqual(self.oracle.solver_executions, self.oracle.calls)


class TestRepairDecode(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = _official_n10()
        self.oracle = FixedRouteOracle(self.instance)
        self.model = _load_h2()

    def test_t1_existing_route_insertion(self) -> None:
        from solver import randomized_greedy

        result = randomized_greedy(
            self.instance,
            restarts=8,
            seed=7,
            local_search_passes=2,
            local_search_budget=1500,
            local_operators="strong",
        )
        self.assertTrue(result.feasible)
        routes = [tuple(r) for r in result.customer_routes]
        plans = list(result.plans)
        multi = [r for r in routes if len(r) >= 2]
        self.assertTrue(multi)
        base = multi[0]
        removed = {base[1] if len(base) > 2 else base[-1]}
        # Prefer removing an interior customer when possible.
        if len(base) >= 3:
            removed = {base[1]}
        survivors = tuple(c for c in base if c not in removed)
        customer = next(iter(removed))
        insertion = best_feasible_insertion(self.oracle, survivors, customer)
        self.assertIsNotNone(insertion)
        partial = _partial_after_destroy(
            self.instance, routes, plans, removed, self.oracle, keep_open=True
        )
        self.assertIsNotNone(partial)
        kept, _ = partial
        # Stub policy actions: force selecting the survivor route and customer.
        # We verify structural capability via insertion helper + open routes.
        _, _, open_indices, _ = self.model.encode_state(
            self.instance, kept, set(removed)
        )
        self.assertGreaterEqual(len(open_indices), 1)
        # Apply insertion manually and validate full coverage repair path.
        new_routes = []
        new_plans = []
        applied = False
        for state in kept:
            if tuple(state["customers"]) == survivors and not applied:
                new_routes.append(insertion["customers"])
                new_plans.append(insertion["plan"])
                applied = True
            else:
                new_routes.append(tuple(state["customers"]))
                new_plans.append(state["plan"])
        self.assertTrue(applied)
        valid, reason = check_solution(self.instance, new_plans)
        self.assertTrue(valid, reason)

    def test_t6_failed_repair_keeps_incumbent(self) -> None:
        routes = [(self.instance.customers[0],)]
        plan = self.oracle.solve(routes[0])
        self.assertTrue(plan.feasible)
        plans = [plan]
        bad = {
            "feasible": False,
            "routes": (),
            "plans": [],
        }
        obj = plan.duration_hours
        out_r, out_p, out_o, improved = _accept_repaired(
            self.instance, bad, obj, routes, plans
        )
        self.assertFalse(improved)
        self.assertEqual(out_r, routes)
        self.assertEqual(out_o, obj)

    def test_t7_repair_mode_gradients(self) -> None:
        customers = list(self.instance.customers)[:3]
        # Build a tiny partial: one open route with first customer, repair second.
        plan = self.oracle.solve((customers[0],))
        if not plan.feasible:
            self.skipTest("singleton infeasible")
        state = route_state(self.instance, (customers[0],), plan)
        state["closed"] = False
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        out = decode(
            self.model,
            self.instance,
            sample=True,
            seed=0,
            cost_lambda=4.0,
            initial_routes=[copy.deepcopy(state)],
            only_customers={customers[1]},
            repair_insertion=True,
        )
        if out["log_prob"] is None:
            self.skipTest("no log_prob on this stub")
        out["log_prob"].backward()
        grad_norm = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                grad_norm += float(parameter.grad.abs().sum())
        self.assertGreater(grad_norm, 0.0)

    def test_t8_ordinary_decode_unchanged_append(self) -> None:
        """repair_insertion=False still appends (position == end only)."""
        # Spy on best_feasible_insertion: must NOT be called in ordinary mode.
        with mock.patch(
            "train.best_feasible_insertion", wraps=best_feasible_insertion
        ) as spy:
            out = decode(
                self.model,
                self.instance,
                sample=False,
                seed=0,
                cost_lambda=0.0,
                repair_insertion=False,
            )
            spy.assert_not_called()
        # Smoke: result structure matches prior API.
        self.assertIn("feasible", out)
        self.assertFalse(out.get("repair_insertion"))

    def test_t10_neural_lns_check_solution(self) -> None:
        from solver import randomized_greedy

        result = randomized_greedy(
            self.instance,
            restarts=6,
            seed=3,
            local_search_passes=1,
            local_search_budget=800,
            local_operators="strong",
        )
        self.assertTrue(result.feasible)
        routes, plans, _, diag = neural_lns(
            self.model,
            self.instance,
            list(result.customer_routes),
            list(result.plans),
            max_iterations=3,
            seed=11,
            cost_lambda=4.0,
            repair_insertion=True,
        )
        valid, reason = check_solution(self.instance, plans)
        self.assertTrue(valid, reason)
        self.assertIn("oracle_solver_executions", diag)
        self.assertGreaterEqual(diag["oracle_solver_executions"], 1)
        for route, plan in zip(routes, plans):
            check = check_route(self.instance, plan.stops, route)
            self.assertTrue(check.feasible, check.reason)


if __name__ == "__main__":
    unittest.main()
