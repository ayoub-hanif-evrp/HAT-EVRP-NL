"""Phase-1 correctness: CSV logging, validation, fingerprints, decomposition."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import (
    FixedRouteOracle,
    RoutePlan,
    check_route,
    check_solution,
    decompose_route,
    route_sequence_fingerprint,
)
from data import generate_synthetic
from train import write_training_csv


class TrainingCsvSchemaTests(unittest.TestCase):
    def test_union_schema_bc_then_rl(self) -> None:
        rows = [
            {
                "stage": "bc",
                "step": 1,
                "loss": 0.5,
                "validation_objective": 20.0,
                "competing_frac": 0.1,
            },
            {
                "stage": "rl",
                "step": 1,
                "loss": 0.2,
                "validation_objective": 19.5,
                "mean_sampled_reward": -19.5,
                "grad_norm": 1.2,
                "competing_frac": "",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.csv"
            write_training_csv(path, rows)
            with path.open(encoding="utf-8", newline="") as handle:
                loaded = list(csv.DictReader(handle))
        self.assertEqual(len(loaded), 2)
        self.assertIn("grad_norm", loaded[0])
        self.assertEqual(loaded[0]["grad_norm"], "")
        self.assertEqual(loaded[1]["grad_norm"], "1.2")
        self.assertEqual(loaded[1]["mean_sampled_reward"], "-19.5")
        self.assertEqual(loaded[0]["competing_frac"], "0.1")


class RouteFingerprintTests(unittest.TestCase):
    def test_vehicle_order_invariant(self) -> None:
        a = [(1, 2, 3), (4, 5)]
        b = [(4, 5), (1, 2, 3)]
        self.assertEqual(route_sequence_fingerprint(a), route_sequence_fingerprint(b))

    def test_customer_order_matters(self) -> None:
        a = [(1, 2, 3)]
        b = [(3, 2, 1)]
        self.assertNotEqual(route_sequence_fingerprint(a), route_sequence_fingerprint(b))


def _official_or_synthetic(n: int = 10, seed: int = 0):
    from data import instance_files, parse_instance

    official = [
        p
        for p in instance_files(ROOT / "dataset")
        if f"c{n}" in p.name.lower() or f"c{n}s" in p.name.lower()
    ]
    if official:
        return parse_instance(official[0])
    return generate_synthetic(n, seed, generator_version="v2c")


class ValidationStrengthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = _official_or_synthetic(10, 7)
        self.oracle = FixedRouteOracle(self.instance)

    def _feasible_plan(self) -> RoutePlan:
        for customer in self.instance.customers:
            plan = self.oracle.solve((customer,))
            if plan.feasible:
                return plan
        plan = self.oracle.solve(tuple(self.instance.customers[:2]))
        self.assertTrue(plan.feasible)
        return plan

    def test_reject_nonfinite_charge_amount(self) -> None:
        plan = self._feasible_plan()
        stops = list(plan.stops)
        # Inject NaN charge on a station stop if present; otherwise at a fake station index.
        injected = False
        for i, (node, amount) in enumerate(stops):
            if self.instance.nodes[node].kind == 2:
                stops[i] = (node, float("nan"))
                injected = True
                break
        if not injected:
            self.skipTest("no charging stop on simple route")
        check = check_route(self.instance, tuple(stops))
        self.assertFalse(check.feasible)
        self.assertIn("nonfinite", check.reason.lower())

    def test_reject_corrupted_stored_duration(self) -> None:
        # Build a full feasible covering solution by solving singles / pairs greedily.
        from solver import randomized_greedy

        result = randomized_greedy(
            self.instance,
            restarts=1,
            seed=0,
            local_search_passes=0,
            local_search_budget=0,
            deterministic_first=True,
        )
        if not result.feasible:
            self.skipTest("greedy infeasible on synthetic")
        plans = list(result.plans)
        bad = RoutePlan(
            True,
            plans[0].duration_hours + 5.0,
            plans[0].stops,
            "corrupted",
        )
        plans[0] = bad
        valid, reason = check_solution(self.instance, plans)
        self.assertFalse(valid)
        self.assertIn("stored duration", reason.lower())

    def test_reject_duplicate_customers_across_routes(self) -> None:
        plan = self._feasible_plan()
        customers = plan.stops and [
            n for n, _ in plan.stops if self.instance.nodes[n].kind == 1
        ]
        if not customers:
            self.skipTest("no customers")
        # Two identical single-customer routes → duplicate coverage.
        plans = [plan, plan]
        valid, reason = check_solution(self.instance, plans)
        self.assertFalse(valid)
        self.assertTrue(
            "multiple routes" in reason.lower() or "coverage" in reason.lower()
        )

    def test_reject_nonfinite_objective_plan(self) -> None:
        plan = self._feasible_plan()
        bad = RoutePlan(True, float("nan"), plan.stops, "")
        valid, reason = check_solution(self.instance, [bad])
        self.assertFalse(valid)
        self.assertIn("nonfinite", reason.lower())


class DecompositionIdentityTests(unittest.TestCase):
    def test_independent_travel_identity(self) -> None:
        instance = _official_or_synthetic(10, 11)
        from solver import randomized_greedy

        result = randomized_greedy(
            instance,
            restarts=1,
            seed=1,
            local_search_passes=0,
            local_search_budget=0,
            deterministic_first=True,
        )
        if not result.feasible:
            self.skipTest("greedy infeasible")
        for plan in result.plans:
            part = decompose_route(instance, plan.stops)
            self.assertTrue(part["feasible"])
            self.assertTrue(part["identity_ok"], part.get("reason"))
            self.assertTrue(math.isfinite(part["travel_h"]))
            rebuilt = part["travel_h"] + part["service_h"] + part["charging_h"]
            self.assertAlmostEqual(rebuilt, part["full_h"], places=6)


class OperatorAttributionTests(unittest.TestCase):
    def test_accepted_credits_responsible_operator(self) -> None:
        from solver_ls import local_search

        instance = generate_synthetic(10, 3, generator_version="v2c")
        oracle = FixedRouteOracle(instance)
        from solver import randomized_greedy

        base = randomized_greedy(
            instance,
            restarts=3,
            seed=5,
            local_search_passes=0,
            local_search_budget=0,
            deterministic_first=True,
        )
        if not base.feasible:
            self.skipTest("greedy infeasible")
        routes = [tuple(r) for r in base.customer_routes]
        plans = list(base.plans)
        before = sum(p.duration_hours for p in plans)
        new_routes, new_plans, improvements, diag = local_search(
            routes,
            plans,
            oracle,
            max_passes=2,
            evaluation_budget=400,
            operators="basic",
        )
        after = sum(p.duration_hours for p in new_plans)
        if improvements == 0:
            self.skipTest("no LS improvement on this seed")
        self.assertLess(after, before)
        accepted_total = sum(s.accepted for s in diag.operators.values())
        self.assertEqual(accepted_total, improvements)


if __name__ == "__main__":
    unittest.main()
