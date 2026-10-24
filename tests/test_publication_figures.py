"""Allowlist and regeneration tests for the main-paper figure set."""

from __future__ import annotations

import csv
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTING = ROOT / "scripts" / "reporting"


class PublicationFigureTests(unittest.TestCase):
    def test_no_bottom_fig_text_in_plot_scripts(self):
        """Figures must not use fig.text / figtext footers for commentary."""
        banned = re.compile(r"\b(?:fig\.text|plt\.figtext)\s*\(")
        for name in (
            "publication_figures.py",
            "bc_rl_full_figures.py",
            "learning_presentation.py",
            "charging_and_reward_figures.py",
            "build_results.py",
        ):
            text = (REPORTING / name).read_text(encoding="utf-8")
            self.assertIsNone(banned.search(text), f"{name} still contains fig.text/figtext")

    def test_fig16_is_categorical_dot_plot(self):
        text = (REPORTING / "bc_rl_full_figures.py").read_text(encoding="utf-8")
        m = re.search(r"def make_fig16\(\).*?(?=\ndef make_fig|\Z)", text, re.S)
        self.assertIsNotNone(m)
        body = m.group(0)
        self.assertIn("ax.scatter(", body)
        self.assertNotIn("ax.bar(", body)
        self.assertIn("categorical", body)

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
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
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
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for r in rows:
            if r.get("stage") != "hybrid_insert" or r.get("feasible") != "1":
                continue
            tot = float(r["travel_h"]) + float(r["service_h"]) + float(r["charging_h"])
            self.assertAlmostEqual(tot, float(r["full_h"]), places=5, msg=r["instance"])

    def test_paper_embeds_main_figures(self):
        paper = (ROOT / "paper_draft.md").read_text(encoding="utf-8")
        for name in (
            "fig01_hat_evrp_nl_architecture.png",
            "fig_validation_protocols.png",
            "fig_validation_learning_curves.png",
            "fig_greedy_generalization.png",
            "fig_validation_seed_effects.png",
            "fig_official_sampled.png",
            "fig_architecture_n10_greedy.png",
            "fig_nonlinear_feasibility.png",
            "fig_quality_vs_compute.png",
            "fig_charging_curves.png",
            "fig_linear_plan_infeasible.png",
            "fig_soc_schedule.png",
            "fig_rl_duration_components.png",
        ):
            self.assertIn(name, paper, f"paper missing embed/reference for {name}")

    def test_paper_captions_hold_interpretive_limits(self):
        paper = (ROOT / "paper_draft.md").read_text(encoding="utf-8")
        self.assertIn("n=80", paper)
        self.assertIn("does not reliably improve", paper)
        self.assertIn("[−0.944, −0.484]", paper)
        self.assertIn("not a better deployment policy than greedy", paper)
        self.assertIn("not used as evidence that REINFORCE", paper)
        self.assertIn("0.0625", paper)
        self.assertIn("Holm", paper)

    def test_central_sign_test_is_not_five_percent(self):
        import csv

        path = ROOT / "results" / "tables" / "final" / "statistical_sensitivity.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["status"] == "central" and row["protocol"] == "sampled_mean"
            ]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(float(row["sign_p_raw"]), 0.0625, places=4)
            self.assertAlmostEqual(float(row["wilcoxon_p_raw"]), 0.0625, places=4)
            self.assertAlmostEqual(float(row["sign_p_holm"]), 0.125, places=4)
            self.assertLess(float(row["t_p_holm"]), 0.05)

    def test_official_sample_file_has_12000_unique_rows(self):
        import sys

        sys.path.insert(0, str(ROOT / "scripts" / "reporting"))
        from final_research_artifacts import validate_official_samples

        audit = validate_official_samples()
        self.assertEqual(audit["rows"], 12000)
        self.assertEqual(len(audit["sha256"]), 64)

    def test_fig15_delta_matches_raw_fairls(self):
        cycle3 = ROOT / "results" / "raw" / "cycle3"
        for n in (10, 20, 40, 80):
            path = cycle3 / f"FINAL_n{n}_fairls.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8", newline="") as handle:
                rows = {r["method"]: r for r in csv.DictReader(handle)}
            if "H2" not in rows or "H4" not in rows:
                continue
            delta = float(rows["H4"]["mean_h"]) - float(rows["H2"]["mean_h"])
            # Sanity: architecture Δ must be finite and small relative to absolute objective.
            self.assertTrue(abs(delta) < 5.0, f"n={n} Δ={delta}")

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
