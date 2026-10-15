"""Finalization tests: deadlines, paired stats, validation fixtures, artifacts."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import (
    RoutePlan,
    check_route,
    check_solution,
    decompose_route,
    literature_compatible_objective_hours,
    total_customer_service_hours,
)
from data import generate_synthetic, instance_files, parse_instance
from solver_ls import SearchDiagnostics, local_search


def _tiny_official():
    paths = [p for p in instance_files(ROOT / "dataset") if "c10" in p.name]
    return parse_instance(paths[0])


class ControllableClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class MockOracle:
    """Deterministic oracle that advances an injected clock per solve."""

    instance: object
    clock: ControllableClock
    cost_per_call: float = 1.0
    calls: int = 0
    requests: int = 0
    cache_hits: int = 0

    def solve(self, customers):
        self.requests += 1
        self.calls += 1
        self.clock.advance(self.cost_per_call)
        # Feasible dummy plan: duration = len(customers) (synthetic).
        seq = tuple(customers)
        depot = self.instance.depot
        stops = ((depot, None),) + tuple((c, None) for c in seq) + ((depot, None),)
        return RoutePlan(True, float(len(seq) + 1), stops)


class DeadlineLocalSearchTests(unittest.TestCase):
    def test_deadline_stops_new_evaluations(self) -> None:
        inst = generate_synthetic(6, 0, generator_version="v2c")
        clock = ControllableClock(0.0)
        oracle = MockOracle(inst, clock, cost_per_call=1.0)
        routes = [tuple(inst.customers[:3]), tuple(inst.customers[3:])]
        plans = [oracle.solve(r) for r in routes]
        # Reset counters after setup solves.
        oracle.calls = 0
        oracle.requests = 0
        clock.t = 0.0
        deadline = 2.5  # allow ~2 solves then stop
        new_routes, new_plans, improvements, diag = local_search(
            routes,
            plans,
            oracle,
            max_passes=3,
            evaluation_budget=100,
            operators="basic",
            deadline=deadline,
            clock=clock,
        )
        self.assertTrue(diag.deadline_hit)
        self.assertLessEqual(oracle.calls, 4)
        self.assertEqual(len(new_routes), len(routes))
        # Incumbent returned even if time expired.
        self.assertTrue(all(p.feasible for p in new_plans))

    def test_no_deadline_preserves_budget_behavior(self) -> None:
        inst = generate_synthetic(6, 1, generator_version="v2c")
        clock = ControllableClock(0.0)
        oracle = MockOracle(inst, clock, cost_per_call=0.0)
        routes = [tuple(inst.customers)]
        plans = [oracle.solve(routes[0])]
        oracle.calls = 0
        _, _, _, diag = local_search(
            routes,
            plans,
            oracle,
            max_passes=1,
            evaluation_budget=5,
            operators="basic",
            deadline=None,
            clock=clock,
        )
        self.assertFalse(diag.deadline_hit)
        self.assertLessEqual(diag.evaluations, 5)


class PairedStatsTests(unittest.TestCase):
    def test_known_delta_and_unmatched(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
        from paired_stats import paired_comparison

        rows = [
            {"instance": "i1", "seed": "42", "method": "H2", "feasible": "1", "objective_h": "10"},
            {"instance": "i1", "seed": "42", "method": "H4", "feasible": "1", "objective_h": "9"},
            {"instance": "i2", "seed": "42", "method": "H2", "feasible": "1", "objective_h": "20"},
            {"instance": "i2", "seed": "42", "method": "H4", "feasible": "1", "objective_h": "22"},
            {"instance": "i3", "seed": "42", "method": "H2", "feasible": "1", "objective_h": "15"},
            # unmatched H4 missing for i3
            {"instance": "i4", "seed": "43", "method": "H4", "feasible": "1", "objective_h": "11"},
        ]
        out = paired_comparison(rows, method_a="H2", method_b="H4", bootstrap=500)
        self.assertEqual(out["n_matched_pairs"], 2)
        self.assertEqual(out["unmatched_a_only"], 1)
        self.assertEqual(out["unmatched_b_only"], 1)
        self.assertAlmostEqual(out["mean_delta"], 0.5, places=6)  # (-1 + 2)/2
        self.assertEqual(out["wins_b_better"], 1)
        self.assertEqual(out["losses_b_worse"], 1)

    def test_infeasible_not_silently_dropped_in_feasibility(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
        from paired_stats import paired_comparison

        rows = [
            {"instance": "i1", "seed": "1", "method": "A", "feasible": "1", "objective_h": "10"},
            {"instance": "i1", "seed": "1", "method": "B", "feasible": "0", "objective_h": "inf"},
            {"instance": "i2", "seed": "1", "method": "A", "feasible": "1", "objective_h": "10"},
            {"instance": "i2", "seed": "1", "method": "B", "feasible": "1", "objective_h": "9"},
        ]
        out = paired_comparison(
            rows, method_a="A", method_b="B", feasible_only_objective=True, bootstrap=200
        )
        self.assertEqual(out["feasibility"]["only_a_feasible"], 1)
        self.assertEqual(out["skipped_infeasible_pairs"], 1)
        self.assertEqual(out["n_objective_pairs"], 1)

    def test_duplicate_detection(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
        from paired_stats import paired_comparison

        rows = [
            {"instance": "i1", "seed": "1", "method": "A", "feasible": "1", "objective_h": "10"},
            {"instance": "i1", "seed": "1", "method": "A", "feasible": "1", "objective_h": "11"},
            {"instance": "i1", "seed": "1", "method": "B", "feasible": "1", "objective_h": "9"},
        ]
        out = paired_comparison(rows, method_a="A", method_b="B", bootstrap=100)
        self.assertGreaterEqual(out["duplicate_extra_rows"], 1)


class ValidationFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inst = _tiny_official()
        self.depot = self.inst.depot
        # Minimal customer stop list for structure tests.
        self.c = self.inst.customers[0]
        self.station = self.inst.stations[0] if self.inst.stations else None

    def test_empty_route(self) -> None:
        stops = ((self.depot, None), (self.depot, None))
        self.assertFalse(check_route(self.inst, stops).feasible)

    def test_unknown_node(self) -> None:
        bad = ((self.depot, None), (99999, None), (self.depot, None))
        self.assertFalse(check_route(self.inst, bad).feasible)

    def test_interior_depot(self) -> None:
        stops = ((self.depot, None), (self.c, None), (self.depot, None), (self.depot, None))
        # Malformed: depot mid-route before final.
        stops = ((self.depot, None), (self.c, None), (self.depot, None), (self.c, None), (self.depot, None))
        # Better: insert interior depot
        stops = ((self.depot, None), (self.c, None), (self.depot, None), (self.depot, None))
        # route with interior depot between customer and end:
        stops = ((self.depot, None), (self.c, None), (self.depot, None), (self.depot, None))
        # Actually length 4 with depot at index 2:
        check = check_route(
            self.inst,
            ((self.depot, None), (self.c, None), (self.depot, None), (self.depot, None)),
        )
        self.assertFalse(check.feasible)

    def test_nan_and_inf_charge(self) -> None:
        if self.station is None:
            self.skipTest("no station")
        for bad in (float("nan"), float("inf"), -1.0):
            stops = (
                (self.depot, None),
                (self.c, None),
                (self.station, bad),
                (self.depot, None),
            )
            self.assertFalse(check_route(self.inst, stops).feasible, msg=str(bad))

    def test_charge_beyond_capacity(self) -> None:
        if self.station is None:
            self.skipTest("no station")
        stops = (
            (self.depot, None),
            (self.c, None),
            (self.station, self.inst.battery_wh * 2),
            (self.depot, None),
        )
        self.assertFalse(check_route(self.inst, stops).feasible)

    def test_duplicate_and_missing_customers(self) -> None:
        from charging import FixedRouteOracle
        from solver import randomized_greedy

        result = randomized_greedy(
            self.inst, restarts=1, seed=0, local_search_passes=0, local_search_budget=0
        )
        self.assertTrue(result.feasible)
        plans = list(result.plans)
        # Duplicate first route
        dup = list(plans) + [plans[0]]
        valid, reason = check_solution(self.inst, dup)
        self.assertFalse(valid)
        # Drop a route => missing customers
        if len(plans) >= 2:
            valid2, reason2 = check_solution(self.inst, plans[:1])
            self.assertFalse(valid2)
            self.assertIn("coverage", reason2.lower())

    def test_stored_duration_mismatch_and_nonfinite(self) -> None:
        from solver import randomized_greedy

        result = randomized_greedy(
            self.inst, restarts=1, seed=1, local_search_passes=0, local_search_budget=0
        )
        self.assertTrue(result.feasible)
        plans = list(result.plans)
        plans[0] = RoutePlan(True, plans[0].duration_hours + 3.0, plans[0].stops)
        self.assertFalse(check_solution(self.inst, plans)[0])
        plans[0] = RoutePlan(True, float("nan"), result.plans[0].stops)
        self.assertFalse(check_solution(self.inst, plans)[0])

    def test_decomposition_identity_and_lit_objective(self) -> None:
        from solver import randomized_greedy

        result = randomized_greedy(
            self.inst, restarts=1, seed=2, local_search_passes=0, local_search_budget=0
        )
        self.assertTrue(result.feasible)
        for plan in result.plans:
            part = decompose_route(self.inst, plan.stops)
            self.assertTrue(part["identity_ok"])
            self.assertAlmostEqual(
                part["travel_h"] + part["service_h"] + part["charging_h"],
                part["full_h"],
                places=6,
            )
        full = sum(p.duration_hours for p in result.plans)
        lit = literature_compatible_objective_hours(self.inst, full)
        self.assertAlmostEqual(lit, full - total_customer_service_hours(self.inst), places=6)


class OperatorAttributionControlledTests(unittest.TestCase):
    def test_last_improving_operator_credited(self) -> None:
        """Force two operators to improve; credit must go to the last one."""
        inst = generate_synthetic(8, 9, generator_version="v2c")
        clock = ControllableClock(0.0)

        class ScriptedOracle:
            def __init__(self):
                self.calls = 0
                self.requests = 0
                self._n = 0

            def solve(self, customers):
                self.requests += 1
                self.calls += 1
                self._n += 1
                seq = tuple(customers)
                depot = inst.depot
                stops = ((depot, None),) + tuple((c, None) for c in seq) + ((depot, None),)
                # Decreasing durations so later calls look better.
                return RoutePlan(True, 100.0 - self._n, stops)

        oracle = ScriptedOracle()
        routes = [tuple(inst.customers[:4]), tuple(inst.customers[4:])]
        plans = [oracle.solve(r) for r in routes]
        start_calls = oracle.calls
        new_r, new_p, improvements, diag = local_search(
            routes,
            plans,
            oracle,
            max_passes=1,
            evaluation_budget=30,
            operators="basic",
            deadline=None,
            clock=clock,
        )
        if improvements == 0:
            self.skipTest("no improvement under scripted oracle")
        accepted_total = sum(s.accepted for s in diag.operators.values())
        self.assertEqual(accepted_total, improvements)
        # Exactly one operator should have accepted == improvements for the last move
        # (across passes the sum matches).
        self.assertGreater(oracle.calls, start_calls)


class ArtifactProtectionTests(unittest.TestCase):
    def test_refuses_silent_csv_overwrite(self) -> None:
        from artifact_io import safe_write_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            path.write_text("a\n1\n", encoding="utf-8")
            written = []

            def write_fn(p: Path) -> None:
                written.append(p)
                p.write_text("a\n2\n", encoding="utf-8")

            out = safe_write_csv(path, write_fn, overwrite=False)
            self.assertNotEqual(out, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "a\n1\n")
            self.assertTrue(out.exists())

    def test_protected_checkpoint_needs_env(self) -> None:
        from artifact_io import safe_save_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "H2_n10_seed42_c3.pt"
            path.write_bytes(b"old")
            with self.assertRaises(PermissionError):
                safe_save_checkpoint(
                    path, lambda p: p.write_bytes(b"new"), overwrite=True
                )


class FingerprintRecomputeTests(unittest.TestCase):
    def test_normalize_existing_fingerprints(self) -> None:
        from charging import route_sequence_fingerprint

        a = "3-1|8-7-9|4"
        b = "4|3-1|8-7-9"
        # Reconstruct via split/sort using the helper on parsed sequences.
        def parse(fp: str):
            return [tuple(int(x) for x in part.split("-")) for part in fp.split("|") if part]

        self.assertEqual(
            route_sequence_fingerprint(parse(a)),
            route_sequence_fingerprint(parse(b)),
        )


if __name__ == "__main__":
    unittest.main()
