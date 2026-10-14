# Results index (publication layer)

This directory separates **presentation** (`figures/`, `tables/`) from **evidence** (`raw/`, `archive/`).

## Current strongest supported results

| Setting | Configuration | Mean full h | Notes |
|---|---|---:|---|
| n=40 | H4 Cycle-3 + ordinary λ=0 + hybrid_insert | **56.370** | Beats hist fair-LS 57.506; single seed |
| n=40 | H2 Cycle-3 + ordinary λ=0 + hybrid_insert | **56.806** | Beats hist fair-LS 57.173 |
| n=10 | H2 Cycle-3 fair-LS | **19.813** | Multi-seed Cycle-3 baseline |

## Do not cite as current best

- INSERTION forced+λ4 n40 hybrid (~63.8 / 66.9 h): construction regression.
- Naive ~43% BKS gaps: wrong objective convention (full vs lit).

## Important limitations

- H4’s better n40 mean in the recovered single-seed experiment does **not** establish a reliable general DeepSets advantage.
- Equal wall-clock neural-search superiority has **not** been established.
- Official benchmark evaluation was revisited repeatedly; treat recent search results as exploratory.
- `frvcpy` is exact for the **fixed-route** charging subproblem, not a proof of global VRP optimality.

## Main-paper figures (exactly 8 PNGs)

See [`../docs/FIGURE_AUDIT.md`](../docs/FIGURE_AUDIT.md).

- [`figures/fig01_architecture.png`](figures/fig01_architecture.png)
- [`figures/fig02_nonlinear_charging_curves.png`](figures/fig02_nonlinear_charging_curves.png)
- [`figures/fig03_charging_model_feasibility.png`](figures/fig03_charging_model_feasibility.png)
- [`figures/fig04_charging_schedule_case.png`](figures/fig04_charging_schedule_case.png)
- [`figures/fig05_objective_decomposition.png`](figures/fig05_objective_decomposition.png)
- [`figures/fig06_architecture_ablation.png`](figures/fig06_architecture_ablation.png)
- [`figures/fig07_learning_and_reward.png`](figures/fig07_learning_and_reward.png)
- [`figures/fig08_quality_compute.png`](figures/fig08_quality_compute.png)

## Supplementary figures

Supporting plots live in [`supplementary/figures/`](supplementary/figures/).
Superseded plots live in [`archive/figures_legacy/`](archive/figures_legacy/).

## Tables (Markdown)

- [`tables/table01_benchmark_overview.md`](tables/table01_benchmark_overview.md)
- [`tables/table02_objective_and_charging.md`](tables/table02_objective_and_charging.md)
- [`tables/table03_main_models.md`](tables/table03_main_models.md)
- [`tables/table04_search_ablation.md`](tables/table04_search_ablation.md)
- [`tables/table05_scalability.md`](tables/table05_scalability.md)
- [`tables/table06_bc_vs_rl.md`](tables/table06_bc_vs_rl.md)
- [`tables/table06_h2_vs_h4.md`](tables/table06_h2_vs_h4.md)
- [`tables/table07_charging_analysis.md`](tables/table07_charging_analysis.md)
- [`tables/table08_bks_comparison.md`](tables/table08_bks_comparison.md)
- [`tables/table09_reproducibility.md`](tables/table09_reproducibility.md)
- [`tables/table10_training_configuration.md`](tables/table10_training_configuration.md)
- [`tables/table11_reward_functions.md`](tables/table11_reward_functions.md)
- [`tables/table12_bc_vs_rl.md`](tables/table12_bc_vs_rl.md)
- [`tables/table13_training_diagnostics.md`](tables/table13_training_diagnostics.md)
- [`tables/table14_neural_vs_search_contribution.md`](tables/table14_neural_vs_search_contribution.md)
- [`tables/table15_nonlinear_vs_linear_charging.md`](tables/table15_nonlinear_vs_linear_charging.md)
- [`tables/table_fig04_schedule_case.md`](tables/table_fig04_schedule_case.md)

## Raw evidence

- `raw/cycle3/` — Cycle-3 fair-LS CSVs
- `raw/nonlinear_charging/` — RQ3 linear vs nonlinear study
- `raw/reward_learning/` — controlled RQ4 BC/RL + reward logs
- `raw/regression_recovery/` — recovered construction + hybrid

## Key reports

- [`NONLINEAR_CHARGING_STUDY.md`](NONLINEAR_CHARGING_STUDY.md)
- [`REWARD_AND_LEARNING_STUDY.md`](REWARD_AND_LEARNING_STUDY.md)
- [`FINAL_REPORT.md`](FINAL_REPORT.md) (Cycle-3 baseline)
- [`../docs/FIGURE_AUDIT.md`](../docs/FIGURE_AUDIT.md)
- [`../docs/TRAINING_AND_REWARD_METHODOLOGY.md`](../docs/TRAINING_AND_REWARD_METHODOLOGY.md)

## Regenerate

```bash
python scripts/reporting/publication_figures.py
python scripts/reporting/build_results.py --all --skip-decomp
```
