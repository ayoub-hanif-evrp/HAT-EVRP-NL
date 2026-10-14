"""Central repository paths for reproducible reporting (no algorithm changes)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SUPPLEMENTARY = RESULTS / "supplementary"
SUPP_FIGURES = SUPPLEMENTARY / "figures"
TABLES = RESULTS / "tables"
RAW = RESULTS / "raw"
ARCHIVE = RESULTS / "archive"
ARCHIVE_FIGURES = ARCHIVE / "figures_legacy"
EXAMPLES = RESULTS / "examples"
SCRIPTS = ROOT / "scripts"
DOCS = ROOT / "docs"
LITERATURE_BKS = RESULTS / "literature_bks_froger2021.csv"

# Final main-paper figure allowlist (sequential, unambiguous).
MAIN_FIGURES: tuple[str, ...] = (
    "fig01_architecture.png",
    "fig02_nonlinear_charging_curves.png",
    "fig03_charging_model_feasibility.png",
    "fig04_charging_schedule_case.png",
    "fig05_objective_decomposition.png",
    "fig06_architecture_ablation.png",
    "fig07_learning_and_reward.png",
    "fig08_quality_compute.png",
)

# Authoritative current numerical sources (post-cleanup).
AUTH_N10_FAIRLS = RAW / "cycle3" / "FINAL_n10_fairls.csv"
AUTH_N40_HIST_FAIRLS = RAW / "cycle3" / "compare_size40_from10_fairls_final_seed42_c3.csv"
AUTH_N40_RECOVERY_PIPE = RAW / "regression_recovery" / "RECOVERY_n40_recovered_pipeline.csv"
AUTH_N40_RECOVERY_FAIR = RAW / "regression_recovery" / "RECOVERY_n40_fair_search.csv"
AUTH_N40_ABLATION = RAW / "regression_recovery" / "RECOVERY_n40_construction_ablation.csv"
AUTH_N40_DECOMP = RAW / "regression_recovery" / "RECOVERY_n40_objective_decomposition.csv"
