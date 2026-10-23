"""Regression tests for piecewise-linear charging time.

The curve is concave: each later segment has a lower energy rate, so a charge
in the high state-of-charge region takes longer than the same energy increment
on the first segment.
"""

from __future__ import annotations

import unittest

from charging import ChargingCurve, charging_time, cumulative_charging_time


def _curve() -> ChargingCurve:
    # 0-10 kWh in 0.5 h, 10-20 kWh in 1.0 h, 20-30 kWh in 1.5 h.
    return ChargingCurve(
        "synthetic",
        times=(0.0, 0.5, 1.5, 3.0),
        energies=(0.0, 10_000.0, 20_000.0, 30_000.0),
    )


class ChargingBreakpointTests(unittest.TestCase):
    def test_partial_charge_inside_one_segment(self):
        curve = _curve()
        # 2 kWh to 8 kWh stays inside the first segment: 6/10 of 0.5 h.
        self.assertAlmostEqual(charging_time(curve, 2_000.0, 8_000.0), 0.3, places=9)

    def test_charge_across_one_breakpoint(self):
        curve = _curve()
        # 8 kWh to 15 kWh: 0.2 of the first segment plus 0.5 of the second.
        self.assertAlmostEqual(charging_time(curve, 8_000.0, 15_000.0), 0.6, places=9)

    def test_charge_across_multiple_breakpoints(self):
        curve = _curve()
        # 5 kWh to 25 kWh crosses both interior breakpoints.
        # First remainder 0.25 h, full second segment 1.0 h, half of the third 0.75 h.
        self.assertAlmostEqual(charging_time(curve, 5_000.0, 25_000.0), 2.0, places=9)
        self.assertAlmostEqual(
            charging_time(curve, 5_000.0, 25_000.0),
            cumulative_charging_time(curve, 25_000.0) - cumulative_charging_time(curve, 5_000.0),
            places=12,
        )

    def test_late_state_of_charge_is_slower_than_the_first_segment(self):
        curve = _curve()
        early = charging_time(curve, 0.0, 5_000.0)
        late = charging_time(curve, 25_000.0, 30_000.0)
        self.assertAlmostEqual(early, 0.25, places=9)
        self.assertAlmostEqual(late, 0.75, places=9)
        self.assertGreater(late, early)

    def test_decreasing_charge_is_rejected(self):
        with self.assertRaises(ValueError):
            charging_time(_curve(), 10_000.0, 9_000.0)


if __name__ == "__main__":
    unittest.main()
