"""Scientific presentation integrity: names, rewards, training logs, links."""

from __future__ import annotations

import ast
import csv
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MethodNameTests(unittest.TestCase):
    def test_display_names_cover_core_keys(self):
        from presentation_names import (
            METHOD_DISPLAY,
            METHOD_DISPLAY_NAMES,
            SEARCH_DISPLAY,
            method_label,
            pair_label,
        )

        self.assertIs(METHOD_DISPLAY_NAMES, METHOD_DISPLAY)
        for key in ("H0", "H0_strong", "H1", "H2", "H3", "H4"):
            self.assertIn(key, METHOD_DISPLAY)
            self.assertNotEqual(method_label(key), key)
            # Short labels must be descriptive — never bare internal codes.
            short = method_label(key, short=True)
            self.assertNotEqual(short, key)
            self.assertNotRegex(short, r"^H\d")
            self.assertFalse(re.search(r"\(H\d", short))
            self.assertFalse(re.search(r"\bH\d\b", method_label(key)))
        for key in ("classical_ruin", "neural_insert", "hybrid_insert", "neural_append"):
            self.assertIn(key, SEARCH_DISPLAY)
        for pair in ("H2-H1", "H4-H2"):
            self.assertFalse(re.search(r"\bH\d\b", pair_label(pair)))

    def test_main_tables_avoid_h_codes_in_method_cells(self):
        tables = ROOT / "results" / "tables"
        for name in (
            "table03_main_models.md",
            "table08_bks_comparison.md",
            "benchmark_protocol_comparison.md",
        ):
            text = (tables / name).read_text(encoding="utf-8")
            # Data rows should not use bare (H2) style display labels.
            self.assertNotRegex(text, r"\(H[0-4]\)")
            self.assertNotRegex(text, r"\| H[0-4] \|")


class RewardFormulaTests(unittest.TestCase):
    def test_train_py_contains_documented_rewards(self):
        text = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn('-100.0', text)
        self.assertIn('-50.0', text)
        self.assertIn('sampled_reward = -sampled["objective"] if sampled["feasible"] else -100.0', text)
        self.assertIn('reward = base_obj - result["objective"]', text)
        self.assertIn("reward = -50.0", text)
        tree = ast.parse(text)
        nums = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                nums.add(float(node.value))
            if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, (int, float))
            ):
                nums.add(-float(node.operand.value))
        self.assertIn(-100.0, nums)
        self.assertIn(-50.0, nums)


class TrainingLogTests(unittest.TestCase):
    def test_c3_logs_have_bc_and_rl_stages(self):
        path = ROOT / "results" / "raw" / "training_logs" / "training_H2_n10_seed42_c3.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        stages = {r["stage"] for r in rows}
        self.assertIn("bc", stages)
        self.assertIn("rl", stages)
        bc = [r for r in rows if r["stage"] == "bc"]
        rl = [r for r in rows if r["stage"] == "rl"]
        self.assertGreaterEqual(len(bc), 2)
        self.assertEqual(len(rl), 1)  # only endpoint logged for Cycle-3

    def test_h4_rl_endpoint_worse_than_bc(self):
        path = ROOT / "results" / "raw" / "training_logs" / "training_H4_n10_seed42_c3.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        bc_end = float([r for r in rows if r["stage"] == "bc"][-1]["validation_objective"])
        rl_end = float([r for r in rows if r["stage"] == "rl"][-1]["validation_objective"])
        self.assertGreater(rl_end, bc_end + 5.0)

    def test_derived_bc_rl_does_not_invent_rewards(self):
        derived = ROOT / "results" / "raw" / "training_logs" / "derived" / "bc_vs_rl_per_seed_c3.csv"
        if not derived.exists():
            self.skipTest("derived not built yet")
        with derived.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r.get("sampled_reward_logged"), "0")


class TableAndFigureTests(unittest.TestCase):
    def test_table08_instance_count_is_20_not_5(self):
        path = ROOT / "results" / "tables" / "table08_bks_comparison.md"
        if not path.exists():
            self.skipTest("table08 missing")
        text = path.read_text(encoding="utf-8")
        self.assertIn("20", text)
        self.assertNotRegex(text, r"Instance count:\s*5\b")
        self.assertIn("N instances", text)

    def test_table12_exists_and_mentions_unavailable_rewards(self):
        path = ROOT / "results" / "tables" / "table12_bc_vs_rl.md"
        if not path.exists():
            self.skipTest("table12 missing")
        text = path.read_text(encoding="utf-8")
        self.assertIn("unavailable", text.lower())
        self.assertIn("Route Transformer", text)

    def test_learning_figures_png_only(self):
        main = ROOT / "results" / "figures"
        supp = ROOT / "results" / "supplementary" / "figures"
        if not main.exists():
            self.skipTest("figures missing")
        for path in main.iterdir():
            if path.is_file():
                self.assertEqual(path.suffix.lower(), ".png", path.name)
        # Main learning figures; RL gradient diagnostics are supplementary
        for name in (
            "fig18_behavior_cloning_learning_curves.png",
            "fig19_bc_vs_rl_checkpoint_quality.png",
            "fig20_reinforce_reward_dynamics.png",
        ):
            self.assertTrue((main / name).exists(), name)
        self.assertTrue((supp / "fig21_rl_training_stability.png").exists())
        self.assertFalse((main / "fig21_rl_training_stability.png").exists())
        # Historical learning plots are supplementary (not in main allowlist)
        for name in (
            "fig11_training_supervision_flow.png",
            "fig12_learning_curves_h1h4.png",
            "fig13_rl_effect_h2_h4.png",
            "fig15_bc_vs_rl_ablation.png",
        ):
            self.assertTrue((supp / name).exists(), f"expected {name} in supplementary/")
            self.assertFalse((main / name).exists(), f"{name} must not be in main figures/")

    def test_paper_links_resolve(self):
        paper = (ROOT / "paper_draft.md").read_text(encoding="utf-8")
        # No obsolete SVG architecture path
        self.assertNotIn("fig_architecture.svg", paper)
        # Linked relative paths that look like repo files
        for match in re.findall(r"`(results/[^`]+)`", paper):
            path = ROOT / match
            if match.endswith("/") or "*" in match:
                continue
            if match.endswith(".csv") or match.endswith(".md") or match.endswith(".png"):
                self.assertTrue(path.exists() or "raw/" in match or "archive/" in match, match)


class DecompositionStillHolds(unittest.TestCase):
    def test_n40_hybrid_means_unchanged(self):
        path = (
            ROOT
            / "results"
            / "raw"
            / "regression_recovery"
            / "RECOVERY_n40_recovered_pipeline.csv"
        )
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for method, expected in (("H2", 56.8055), ("H4", 56.3700)):
            objs = [
                float(r["objective_h"])
                for r in rows
                if r["method"] == method
                and r["stage"] == "hybrid_insert"
                and r["feasible"] == "1"
            ]
            self.assertAlmostEqual(sum(objs) / len(objs), expected, places=3)


if __name__ == "__main__":
    unittest.main()
