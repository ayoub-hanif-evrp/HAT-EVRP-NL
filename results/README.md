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

## Figures (PNG only)

- [`figures/fig01_architecture.png`](figures/fig01_architecture.png)
- [`figures/fig02_problem_example.png`](figures/fig02_problem_example.png)
- [`figures/fig03_charging_curves.png`](figures/fig03_charging_curves.png)
- [`figures/fig04_objective_decomposition_n40.png`](figures/fig04_objective_decomposition_n40.png)
- [`figures/fig05_soc_charging_trace.png`](figures/fig05_soc_charging_trace.png)
- [`figures/fig06_main_benchmark_n10_fairls.png`](figures/fig06_main_benchmark_n10_fairls.png)
- [`figures/fig07_scaling_fairls.png`](figures/fig07_scaling_fairls.png)
- [`figures/fig08_search_ablation_n40_recovered.png`](figures/fig08_search_ablation_n40_recovered.png)
- [`figures/fig09_h2_vs_h4_n40_hybrid.png`](figures/fig09_h2_vs_h4_n40_hybrid.png)
- [`figures/fig10_charging_ablation_feasibility.png`](figures/fig10_charging_ablation_feasibility.png)

## Tables (Markdown)

- [`tables/table01_benchmark_overview.md`](tables/table01_benchmark_overview.md)
- [`tables/table02_objective_and_charging.md`](tables/table02_objective_and_charging.md)
- [`tables/table03_main_models.md`](tables/table03_main_models.md)
- [`tables/table04_search_ablation.md`](tables/table04_search_ablation.md)
- [`tables/table05_scalability.md`](tables/table05_scalability.md)
- [`tables/table06_h2_vs_h4.md`](tables/table06_h2_vs_h4.md)
- [`tables/table07_charging_analysis.md`](tables/table07_charging_analysis.md)
- [`tables/table08_bks_comparison.md`](tables/table08_bks_comparison.md)
- [`tables/table09_reproducibility.md`](tables/table09_reproducibility.md)

## Raw evidence

- `raw/cycle3/` — Cycle-3 fair-LS CSVs
- `raw/regression_recovery/` — recovered construction + hybrid
- `raw/insertion_repair/` — invalid forced-start insertion runs (kept as evidence)
- `archive/reports/` — historical Markdown reports

## Key reports

- [`REGRESSION_RECOVERY_REPORT.md`](REGRESSION_RECOVERY_REPORT.md)
- [`FINAL_REPORT.md`](FINAL_REPORT.md) (Cycle-3 baseline)
- [`INSERTION_REPAIR_REPORT.md`](INSERTION_REPAIR_REPORT.md)

## Regenerate

```bash
python scripts/reporting/build_results.py --all
```
