"""Tests for linear charging surrogates and true-physics re-evaluation."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SurrogateDefinitionTests(unittest.TestCase):
    def test_secant_and_early_rate_definitions(self):
        from charging_surrogates import (
            instance_with_charging_model,
            linear_early_rate_curve,
            linear_secant_curve,
            surrogate_rate_wh_per_hour,
        )
        from data import instance_files, parse_instance

        path = next(p for p in instance_files(ROOT / "dataset") if "c10" in p.name)
        inst = parse_instance(path)
        self.assertTrue(inst.curves)
        for stype, true in inst.curves.items():
            sec = linear_secant_curve(true)
            early = linear_early_rate_curve(true)
            self.assertEqual(sec.energies[0], 0.0)
            self.assertEqual(sec.energies[-1], true.energies[-1])
            self.assertEqual(len(sec.times), 2)
            # Early rate >= secant rate for concave (non-decreasing slope decrease) curves
            self.assertGreaterEqual(
                surrogate_rate_wh_per_hour(early),
                surrogate_rate_wh_per_hour(sec) - 1e-9,
            )
            # Surrogate instance keeps topology
            lin = instance_with_charging_model(inst, "LINEAR-SECANT")
            self.assertEqual(lin.customers, inst.customers)
            self.assertEqual(lin.battery_wh, inst.battery_wh)
            self.assertNotEqual(lin.curves[stype].times, true.times)

    def test_xml_unchanged_by_surrogate(self):
        import hashlib

        from charging_surrogates import instance_with_charging_model
        from data import instance_files, parse_instance

        path = next(p for p in instance_files(ROOT / "dataset") if "c10" in p.name)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        inst = parse_instance(path)
        _ = instance_with_charging_model(inst, "LINEAR-EARLY-RATE")
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_true_physics_can_reject_linear_plan_amounts(self):
        """Sanity: charging_time under true NL >= early-rate linear for same q_in,q_out near full."""
        from charging import charging_time
        from charging_surrogates import linear_early_rate_curve
        from data import instance_files, parse_instance

        inst = None
        for path in instance_files(ROOT / "dataset"):
            candidate = parse_instance(path)
            if "fast" in candidate.curves and len(candidate.curves["fast"].times) >= 3:
                inst = candidate
                break
        if inst is None:
            self.skipTest("no multi-breakpoint fast curve")
        true = inst.curves["fast"]
        early = linear_early_rate_curve(true)
        q_in = 0.85 * inst.battery_wh
        q_out = 0.98 * inst.battery_wh
        t_true = charging_time(true, q_in, q_out)
        t_early = charging_time(early, q_in, q_out)
        # Concave charging: late increments slower than early-rate linear ⇒ true >= early
        self.assertGreaterEqual(t_true + 1e-9, t_early)


if __name__ == "__main__":
    unittest.main()
