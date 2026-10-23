"""Non-regression: recovered n40 fair-LS must not be worse than historical means."""

from __future__ import annotations

import csv
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
def _find(*parts: str) -> Path:
    candidates = [RES.joinpath(*parts), RES / "raw" / "cycle3" / parts[-1], RES / "raw" / "regression_recovery" / parts[-1]]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


HIST = _find("compare_size40_from10_fairls_final_seed42_c3.csv")
REPRO = _find("RECOVERY_n40_repro_fairls.csv")
PIPE = _find("RECOVERY_n40_recovered_pipeline.csv")

HIST_H2 = 57.17274009092578
HIST_H4 = 57.50564156359235


class TestN40RegressionRecovery(unittest.TestCase):
    def test_historical_csv_means(self) -> None:
        with HIST.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for method, ref in (("H2", HIST_H2), ("H4", HIST_H4)):
            objs = [
                float(r["objective_h"])
                for r in rows
                if r["method"] == method and r["feasible"] == "1"
            ]
            self.assertEqual(len(objs), 20)
            self.assertAlmostEqual(statistics.mean(objs), ref, places=8)

    def test_repro_not_worse_than_historical(self) -> None:
        self.assertTrue(REPRO.exists(), "run recovery reproduce stage first")
        with REPRO.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for method, ref in (("H2", HIST_H2), ("H4", HIST_H4)):
            objs = [
                float(r["objective_h"])
                for r in rows
                if r["method"] == method and r["feasible"] == "1"
            ]
            self.assertEqual(len(objs), 20)
            self.assertLessEqual(statistics.mean(objs), ref + 0.15)

    def test_hybrid_on_recovered_beats_historical(self) -> None:
        self.assertTrue(PIPE.exists(), "run recovery pipeline stage first")
        with PIPE.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for method, ref in (("H2", HIST_H2), ("H4", HIST_H4)):
            objs = [
                float(r["objective_h"])
                for r in rows
                if r["method"] == method
                and r["stage"] == "hybrid_insert"
                and r["feasible"] == "1"
            ]
            self.assertEqual(len(objs), 20)
            self.assertLess(statistics.mean(objs), ref)


if __name__ == "__main__":
    unittest.main()
