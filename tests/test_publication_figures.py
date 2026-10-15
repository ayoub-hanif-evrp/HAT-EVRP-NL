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
        self.assertGreaterEqual(len(present), 16)
        self.assertEqual(len(present), len(MAIN_FIGURES))
        self.assertGreaterEqual(len(MAIN_FIGURES), 16)

    def test_png_only_and_openable(self):
        from PIL import Image

        from reporting_paths import FIGURES, MAIN_FIGURES

        for name in MAIN_FIGURES:
            path = FIGURES / name
            self.assertTrue(path.exists(), name)
            self.assertEqual(path.suffix.lower(), ".png")
            with Image.open(path) as im:
                w, h = im.size
                self.assertGreater(w, 200, name)
                self.assertGreater(h, 150, name)
                self.assertGreaterEqual(im.info.get("dpi", (300, 300))[0], 72)

    def test_obsolete_eight_set_not_in_main(self):
        from reporting_paths import FIGURES

        banned = {
            "fig01_architecture.png",
            "fig06_architecture_ablation.png",
            "fig07_learning_and_reward.png",
            "fig08_quality_compute.png",
            "fig_main03d_schedule_comparison_example.png",
        }
        present = {p.name for p in FIGURES.glob("*.png")}
        self.assertFalse(present & banned)

    def test_fig06_counts_match_raw(self):
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

    def test_decomp_identity(self):
        import csv

        path = ROOT / "results" / "raw" / "regression_recovery" / "RECOVERY_n40_objective_decomposition.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        for r in rows:
            if r.get("stage") != "hybrid_insert" or r.get("feasible") != "1":
                continue
            tot = float(r["travel_h"]) + float(r["service_h"]) + float(r["charging_h"])
            self.assertAlmostEqual(tot, float(r["full_h"]), places=5, msg=r["instance"])

    def test_paper_embeds_main_figures(self):
        import re

        from reporting_paths import MAIN_FIGURES

        paper = (ROOT / "paper_draft.md").read_text(encoding="utf-8")
        for name in MAIN_FIGURES:
            self.assertIn(name, paper, f"paper missing embed/reference for {name}")
        # caption markers
        self.assertGreaterEqual(len(re.findall(r"\*\*Figure \d+\.", paper)), 16)

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
