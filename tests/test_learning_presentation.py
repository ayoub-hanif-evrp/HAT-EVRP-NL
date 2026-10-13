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
        from presentation_names import METHOD_DISPLAY, SEARCH_DISPLAY, method_label

        for key in ("H0", "H0_strong", "H1", "H2", "H3", "H4"):
            self.assertIn(key, METHOD_DISPLAY)
            self.assertNotEqual(method_label(key), key)
            self.assertIn(key, method_label(key, short=True))
        for key in ("classical_ruin", "neural_insert", "hybrid_insert", "neural_append"):
            self.assertIn(key, SEARCH_DISPLAY)


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
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        stages = {r["stage"] for r in rows}
        self.assertIn("bc", stages)
        self.assertIn("rl", stages)
        bc = [r for r in rows if r["stage"] == "bc"]
        rl = [r for r in rows if r["stage"] == "rl"]
        self.assertGreaterEqual(len(bc), 2)
        self.assertEqual(len(rl), 1)  # only endpoint logged for Cycle-3

    def test_h4_rl_endpoint_worse_than_bc(self):
        path = ROOT / "results" / "raw" / "training_logs" / "training_H4_n10_seed42_c3.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        bc_end = float([r for r in rows if r["stage"] == "bc"][-1]["validation_objective"])
        rl_end = float([r for r in rows if r["stage"] == "rl"][-1]["validation_objective"])
        self.assertGreater(rl_end, bc_end + 5.0)

    def test_derived_bc_rl_does_not_invent_rewards(self):
        derived = ROOT / "results" / "raw" / "training_logs" / "derived" / "bc_vs_rl_per_seed_c3.csv"
        if not derived.exists():
            self.skipTest("derived not built yet")
        rows = list(csv.DictReader(derived.open(encoding="utf-8")))
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
        self.assertIn("History Transformer", text)

    def test_learning_figures_png_only(self):
        figures = ROOT / "results" / "figures"
        if not figures.exists():
            self.skipTest("figures missing")
        for path in figures.iterdir():
            self.assertEqual(path.suffix.lower(), ".png", path.name)
        for name in (
            "fig11_training_supervision_flow.png",
            "fig12_learning_curves_h1h4.png",
            "fig13_rl_effect_h2_h4.png",
            "fig14_reward_dynamics_unavailable.png",
            "fig15_bc_vs_rl_ablation.png",
        ):
            self.assertTrue((figures / name).exists(), name)

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
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
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
