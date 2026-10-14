"""Allowlist and regeneration tests for the main-paper figure set."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PublicationFigureTests(unittest.TestCase):
    def test_main_allowlist_exact(self):
        from reporting_paths import FIGURES, MAIN_FIGURES

        present = sorted(p.name for p in FIGURES.glob("*.png"))
        self.assertEqual(present, sorted(MAIN_FIGURES))
        self.assertEqual(len(present), 8)

    def test_obsolete_names_not_in_main(self):
        from reporting_paths import FIGURES

        banned = {
            "fig_main03d_schedule_comparison_example.png",
            "fig_main03_linear_plan_true_feasibility.png",
            "fig_main03c_e2e_true_feasibility.png",
            "fig_main07_reward_dynamics.png",
            "fig06_main_benchmark_n10_fairls.png",
            "fig17_history_deepsets_contribution.png",
            "fig08_search_ablation_n40_recovered.png",
            "fig14_reward_dynamics_unavailable.png",
        }
        present = {p.name for p in FIGURES.glob("*.png")}
        self.assertFalse(present & banned)

    def test_fig03_counts_match_raw(self):
        import csv
        from collections import defaultdict

        path = ROOT / "results" / "raw" / "nonlinear_charging" / "fixed_route_charging_comparison.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        by = defaultdict(lambda: {"true": 0, "n": 0})
        for r in rows:
            if not r["method"].startswith("plan_"):
                continue
            key = (int(r["size"]), r["charging_model"])
            by[key]["n"] += 1
            by[key]["true"] += int(r["feasible_under_true_physics"])
        self.assertEqual(by[(40, "TRUE-NL")]["true"], 20)
        self.assertEqual(by[(40, "LINEAR-SECANT")]["true"], 11)
        self.assertEqual(by[(40, "LINEAR-EARLY-RATE")]["true"], 11)

    def test_schedule_case_table_exists(self):
        path = ROOT / "results" / "tables" / "table_fig04_schedule_case.md"
        self.assertTrue(path.exists(), path)
        text = path.read_text(encoding="utf-8")
        self.assertIn("tc0c10s2cf1", text)
        self.assertIn("Duration violation", text)

    def test_regeneration_keeps_allowlist(self):
        import sys

        sys.path.insert(0, str(ROOT / "scripts" / "reporting"))
        from publication_figures import build_all_main_figures
        from reporting_paths import FIGURES, MAIN_FIGURES

        build_all_main_figures()
        present = sorted(p.name for p in FIGURES.glob("*.png"))
        self.assertEqual(present, sorted(MAIN_FIGURES))


if __name__ == "__main__":
    unittest.main()
