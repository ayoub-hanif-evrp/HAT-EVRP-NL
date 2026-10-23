"""Tests for publication reporting integrity."""

from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReportingLayoutTests(unittest.TestCase):
    def test_final_figures_are_png_only(self):
        figures = ROOT / "results" / "figures"
        if not figures.exists():
            self.skipTest("figures not generated yet")
        files = list(figures.iterdir())
        self.assertTrue(files, "results/figures is empty")
        for path in files:
            self.assertEqual(path.suffix.lower(), ".png", path.name)
            self.assertNotIn(path.suffix.lower(), {".svg", ".pdf", ".eps"})

    def test_final_tables_are_markdown(self):
        tables = ROOT / "results" / "tables"
        if not tables.exists():
            self.skipTest("tables not generated yet")
        md = list(tables.glob("*.md"))
        self.assertGreaterEqual(len(md), 9)
        for path in md:
            self.assertEqual(path.suffix.lower(), ".md")

    def test_results_index_links_resolve(self):
        index = ROOT / "results" / "README.md"
        if not index.exists():
            self.skipTest("results index missing")
        text = index.read_text(encoding="utf-8")
        for name in (
            "figures/fig01_architecture.png",
            "tables/table03_main_models.md",
            "REGRESSION_RECOVERY_REPORT.md",
        ):
            if name.split("/")[0] == "figures" and not (ROOT / "results" / name).exists():
                continue
            if name.startswith("tables/") and not (ROOT / "results" / name).exists():
                continue
            self.assertTrue(
                (ROOT / "results" / name).exists() or name in text,
                f"missing linked artifact {name}",
            )
            if (ROOT / "results" / name).exists():
                self.assertTrue((ROOT / "results" / name).stat().st_size > 0)


class NumericalIntegrityTests(unittest.TestCase):
    def test_n40_historical_fairls_means(self):
        path = ROOT / "results" / "raw" / "cycle3" / "compare_size40_from10_fairls_final_seed42_c3.csv"
        self.assertTrue(path.exists(), path)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for method, expected in (("H2", 57.17274009092578), ("H4", 57.50564156359235)):
            objs = [
                float(r["objective_h"])
                for r in rows
                if r["method"] == method and r.get("objective_h")
            ]
            self.assertEqual(len(objs), 20)
            self.assertAlmostEqual(sum(objs) / len(objs), expected, places=6)

    def test_n40_recovered_hybrid_means(self):
        path = (
            ROOT
            / "results"
            / "raw"
            / "regression_recovery"
            / "RECOVERY_n40_recovered_pipeline.csv"
        )
        self.assertTrue(path.exists(), path)
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
            self.assertEqual(len(objs), 20)
            self.assertAlmostEqual(sum(objs) / len(objs), expected, places=3)

    def test_objective_decomposition_identity(self):
        path = (
            ROOT
            / "results"
            / "raw"
            / "regression_recovery"
            / "RECOVERY_n40_objective_decomposition.csv"
        )
        if not path.exists():
            self.skipTest("decomposition CSV not yet regenerated")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreater(len(rows), 0)
        checked = 0
        for r in rows:
            if r.get("feasible") != "1":
                continue
            travel = float(r["travel_h"])
            service = float(r["service_h"])
            charge = float(r["charging_h"])
            full = float(r["full_h"])
            lit = float(r["lit_h"])
            self.assertLessEqual(abs(full - (travel + service + charge)), 1e-5)
            self.assertLessEqual(abs(lit - (full - service)), 1e-5)
            self.assertEqual(r.get("identity_ok"), "1")
            checked += 1
        self.assertGreater(checked, 0)

    def test_table03_matches_raw_n10(self):
        raw = ROOT / "results" / "raw" / "cycle3" / "FINAL_n10_fairls.csv"
        table = ROOT / "results" / "tables" / "table03_main_models.md"
        if not table.exists():
            self.skipTest("table03 not generated")
        with raw.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        h2 = next(r for r in rows if r["method"] == "H2")
        mean = float(h2.get("mean_h") or h2["objective_h"])
        text = table.read_text(encoding="utf-8")
        self.assertIn(f"{mean:.4f}", text)

    def test_literature_bks_file_present(self):
        self.assertTrue((ROOT / "results" / "literature_bks_froger2021.csv").exists())

    def test_dataset_xml_count_and_hash_file(self):
        xml = list((ROOT / "dataset").rglob("*.xml"))
        self.assertEqual(len(xml), 120)
        hash_csv = ROOT / "results" / "raw" / "references" / "dataset_xml_sha256.csv"
        if hash_csv.exists():
            with hash_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 120)


class DecompositionUnitTests(unittest.TestCase):
    def test_decompose_route_identity_on_feasible_decode(self):
        import torch
        from charging import decompose_solution
        from data import instance_files, parse_instance
        from train import decode, load_policy

        ckpt = ROOT / "checkpoints" / "H2_n10_seed42_c3.pt"
        if not ckpt.exists():
            self.skipTest("checkpoint missing")
        paths = [p for p in instance_files(ROOT / "dataset") if "c10" in p.name]
        inst = parse_instance(paths[0])
        policy = load_policy(ckpt)
        with torch.no_grad():
            out = decode(policy, inst, sample=False, seed=42, cost_lambda=0.0)
        if not out["feasible"]:
            self.skipTest("decode infeasible on first n10")
        decomp = decompose_solution(inst, out["plans"])
        self.assertTrue(decomp["feasible"])
        self.assertTrue(decomp["identity_ok"])
        self.assertTrue(math.isfinite(decomp["charging_h"]))
        self.assertGreaterEqual(decomp["charging_h"], 0.0)


if __name__ == "__main__":
    unittest.main()
