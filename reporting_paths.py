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

# Final main-paper figure allowlist (~21 sequential PNGs).
MAIN_FIGURES: tuple[str, ...] = (
    "fig01_hat_evrp_nl_architecture.png",
    "fig02_montoya_instance_layouts.png",
    "fig03_feasible_ev_routes.png",
    "fig04_nonlinear_charging_curves.png",
    "fig05_charging_time_vs_soc.png",
    "fig06_fixed_route_charging_feasibility.png",
    "fig07_end_to_end_charging_feasibility.png",
    "fig08_soc_charging_schedule.png",
    "fig09_linear_charging_estimation_error.png",
    "fig10_station_usage_and_energy.png",
    "fig11_objective_decomposition.png",
    "fig12_neural_architecture_ablation.png",
    "fig13_history_deepsets_paired_effect.png",
    "fig14_literature_bks_gap.png",
    "fig15_generalization_scalability.png",
    "fig16_search_improvement_stages.png",
    "fig17_quality_vs_compute.png",
    "fig18_behavior_cloning_learning_curves.png",
    "fig19_bc_vs_rl_checkpoint_quality.png",
    "fig20_reinforce_reward_dynamics.png",
    "fig21_rl_training_stability.png",
)

# Authoritative current numerical sources (post-cleanup).
AUTH_N10_FAIRLS = RAW / "cycle3" / "FINAL_n10_fairls.csv"
AUTH_N40_HIST_FAIRLS = RAW / "cycle3" / "compare_size40_from10_fairls_final_seed42_c3.csv"
AUTH_N40_RECOVERY_PIPE = RAW / "regression_recovery" / "RECOVERY_n40_recovered_pipeline.csv"
AUTH_N40_RECOVERY_FAIR = RAW / "regression_recovery" / "RECOVERY_n40_fair_search.csv"
AUTH_N40_ABLATION = RAW / "regression_recovery" / "RECOVERY_n40_construction_ablation.csv"
AUTH_N40_DECOMP = RAW / "regression_recovery" / "RECOVERY_n40_objective_decomposition.csv"
AUTH_BKS_GAPS = RAW / "cost_aware" / "IMPROVEMENT_bks_gaps_corrected.csv"
SOLUTIONS = RAW / "regression_recovery" / "solutions"
NL_RAW = RAW / "nonlinear_charging"
RL_RAW = RAW / "reward_learning"
TRAIN_LOGS = RAW / "training_logs"
