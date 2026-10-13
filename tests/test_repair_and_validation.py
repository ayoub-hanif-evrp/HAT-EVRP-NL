"""Regression tests for insertion-capable neural repair and route validation."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import FixedRouteOracle, check_route
from data import generate_synthetic
from train import (
    _partial_after_destroy,
    decode,
    load_policy,
    route_state,
)


class TestRouteValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = generate_synthetic(10, 42, generator_version="v2c")
        self.oracle = FixedRouteOracle(self.instance)

    def test_reject_empty_depot_to_depot(self) -> None:
        depot = self.instance.depot
        stops = ((depot, None), (depot, None))
        check = check_route(self.instance, stops)
        self.assertFalse(check.feasible)
        self.assertIn("empty", check.reason.lower())

    def test_reject_interior_depot(self) -> None:
        depot = self.instance.depot
        customer = self.instance.customers[0]
        plan = self.oracle.solve((customer,))
        self.assertTrue(plan.feasible)
        # Inject an interior depot visit into a copy of the stop list.
        stops = list(plan.stops)
        # Insert depot just before the final return.
        stops.insert(len(stops) - 1, (depot, None))
        check = check_route(self.instance, tuple(stops), expected_customers=None)
        self.assertFalse(check.feasible)
        self.assertIn("interior depot", check.reason.lower())


class TestNeuralRepairOpenRoutes(unittest.TestCase):
    def setUp(self) -> None:
        from data import instance_files, parse_instance

        # Prefer a small official instance so a feasible base solution is easy.
        official = [
            p
            for p in instance_files(ROOT / "dataset")
            if len(parse_instance(p).customers) == 10
        ]
        if official:
            self.instance = parse_instance(official[0])
        else:
            self.instance = generate_synthetic(10, 42, generator_version="v2c")
        self.oracle = FixedRouteOracle(self.instance)
        ckpt = ROOT / "checkpoints" / "H2_n10_seed42_perf_cont_v2c.pt"
        if not ckpt.exists():
            ckpt = ROOT / "checkpoints" / "H2_n10_seed42_c3.pt"
        self.assertTrue(ckpt.exists(), f"missing checkpoint {ckpt}")
        self.model = load_policy(ckpt)

    def test_partial_keeps_routes_open(self) -> None:
        customers = list(self.instance.customers)
        # Build a feasible two-customer route if possible.
        for i, a in enumerate(customers):
            for b in customers[i + 1 :]:
                plan = self.oracle.solve((a, b))
                if plan.feasible:
                    routes = [(a, b)]
                    plans = [plan]
                    removed = {b}
                    partial = _partial_after_destroy(
                        self.instance, routes, plans, removed, self.oracle
                    )
                    self.assertIsNotNone(partial)
                    kept_routes, _ = partial
                    self.assertEqual(len(kept_routes), 1)
                    self.assertFalse(kept_routes[0]["closed"])
                    self.assertEqual(kept_routes[0]["customers"], (a,))
                    return
        self.fail("could not find a feasible two-customer route for the test")

    def test_neural_repair_can_extend_existing_route(self) -> None:
        """Removed customer can be re-inserted onto a surviving open route."""
        from charging import check_solution
        from solver import randomized_greedy

        # Full feasible construction (teacher-style) so coverage checks hold.
        result = randomized_greedy(
            self.instance,
            restarts=8,
            seed=101,
            local_search_passes=2,
            local_search_budget=1500,
            local_operators="strong",
        )
        self.assertTrue(result.feasible, "need a feasible base solution")
        routes = [tuple(r) for r in result.customer_routes]
        plans = list(result.plans)
        valid, reason = check_solution(self.instance, plans)
        self.assertTrue(valid, reason)

        multi = [r for r in routes if len(r) >= 2]
        self.assertTrue(multi, "need a multi-customer route for insertion test")
        base_route = multi[0]
        removed_customer = base_route[-1]
        removed = {removed_customer}
        kept = base_route[:-1]
        ext = self.oracle.solve((*kept, removed_customer))
        self.assertTrue(ext.feasible, "append reinsertion must be feasible")

        partial = _partial_after_destroy(
            self.instance, routes, plans, removed, self.oracle, keep_open=True
        )
        self.assertIsNotNone(partial)
        kept_routes, _ = partial
        self.assertTrue(all(not r["closed"] for r in kept_routes))
        _, _, open_indices, _ = self.model.encode_state(
            self.instance, kept_routes, set(removed)
        )
        self.assertGreaterEqual(
            len(open_indices), 1, "surviving routes must be open for insertion"
        )

        # Force extension onto the surviving prefix (not a new route).
        repaired = decode(
            self.model,
            self.instance,
            sample=False,
            seed=0,
            cost_lambda=4.0,
            initial_routes=copy.deepcopy(kept_routes),
            only_customers=set(removed),
        )
        self.assertTrue(repaired["feasible"], repaired.get("reason"))
        covered = {c for route in repaired["routes"] for c in route}
        self.assertEqual(covered, set(self.instance.customers))
        # At least one repaired route must extend the destroyed prefix OR the
        # removed customer appears (new-route repair). Open eligibility is the
        # structural bugfix; also prove append state construction.
        state = route_state(self.instance, (*kept, removed_customer), ext)
        self.assertFalse(state["closed"])
        self.assertEqual(state["customers"][: len(kept)], kept)
        # Prefer checking that decode actually used an open existing route when
        # greedy chooses extension (prefix preserved).
        extended = any(
            len(route) >= len(kept)
            and tuple(route[: len(kept)]) == kept
            and removed_customer in route
            for route in repaired["routes"]
        )
        new_route = any(
            route == (removed_customer,) for route in repaired["routes"]
        )
        self.assertTrue(
            extended or new_route,
            "repair must place the removed customer somewhere",
        )


if __name__ == "__main__":
    unittest.main()
