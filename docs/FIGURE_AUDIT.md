# Figure Audit — HAT-EVRP-NL

**Current main set: 21 PNGs** (eight-figure limit cancelled).  
Catalog: [`FIGURE_CATALOG.md`](FIGURE_CATALOG.md).  
Generator / allowlist: `scripts/reporting/publication_figures.py`, `MAIN_FIGURES` in `reporting_paths.py`.

## Layout

| Directory | Role |
|---|---|
| `results/figures/` | Final main-paper PNGs only (21) |
| `results/supplementary/figures/` | Supporting + contact sheet |
| `results/archive/figures_legacy/` | Superseded (including prior 8-figure set) |

## Disposition of the previous 8-figure set

| Old file | Status |
|---|---|
| `fig01_architecture.png` | Redesigned → `fig01_hat_evrp_nl_architecture.png` |
| `fig02_nonlinear_charging_curves.png` | Redesigned → `fig04_nonlinear_charging_curves.png` |
| `fig03_charging_model_feasibility.png` | Split → `fig06_*` + `fig07_*` |
| `fig04_charging_schedule_case.png` | Redesigned → `fig08_soc_charging_schedule.png` |
| `fig05_objective_decomposition.png` | Redesigned → `fig11_objective_decomposition.png` |
| `fig06_architecture_ablation.png` | Redesigned → `fig12_neural_architecture_ablation.png` |
| `fig07_learning_and_reward.png` | Split → `fig18`–`fig21` |
| `fig08_quality_compute.png` | Redesigned → `fig17_quality_vs_compute.png` |

## Regeneration

```bash
python scripts/reporting/publication_figures.py
python scripts/reporting/build_results.py --all --skip-decomp
```

Legacy builders write only to `supplementary/`. `enforce_figure_layout()` keeps `results/figures/` = `MAIN_FIGURES`.
