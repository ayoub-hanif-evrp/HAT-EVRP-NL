# Figure Catalog — HAT-EVRP-NL

Contact sheet: `results/supplementary/figures/figure_contact_sheet.png`  
Allowlist: `MAIN_FIGURES` in `reporting_paths.py` (20 main PNGs)  
Supplementary: `SUPPLEMENTARY_FIGURES` (fig21 diagnostics)  
Generator: `scripts/reporting/publication_figures.py`  
Audit: `results/SCIENTIFIC_RESULTS_AUDIT.md` · Quality: `docs/FIGURE_QUALITY_REVIEW.md`

| # | File | Question | Raw source | N / protocol | Takeaway | Caveats |
|---|---|---|---|---|---|---|
| 01 | `fig01_hat_evrp_nl_architecture.png` | Architecture | Schematic | — | Neural vs frvcpy vs checker | Schematic |
| 02 | `fig02_montoya_instance_layouts.png` | Geography | Official XML | n=20×3 | Spatial diversity | Family prefixes |
| 03 | `fig03_feasible_ev_routes.png` | Solved routes | solutions JSON | n=40 hybrid | Real routes+charges | One instance |
| 04 | `fig04_nonlinear_charging_curves.png` | NL curves | XML breakpoints | techs | Late-SOC slowdown | — |
| 05 | `fig05_charging_time_vs_soc.png` | Marginal time | PWL slopes | techs | min/kWh rises w/ SOC | — |
| 06 | `fig06_fixed_route_charging_feasibility.png` | Fixed-route feas. | fixed_route CSV | 20/size | Linear fails physics | plan_* rows |
| 07 | `fig07_end_to_end_charging_feasibility.png` | E2E feas. | e2e CSVs | 20/size | Routing changes too | Frozen H2 |
| 08 | `fig08_soc_charging_schedule.png` | SOC case | schedules JSON | one route | EARLY vs Tmax | Case study |
| 09 | `fig09_linear_charging_estimation_error.png` | Time error | fixed_route CSV | recorded | EARLY bias | Selected subset |
| 10 | `fig10_station_usage_and_energy.png` | frvcpy choices | n40 decomp | 20 hybrid | Visits/energy/time | 3 panels |
| 11 | `fig11_objective_decomposition.png` | Obj. parts | n40 decomp | 20 hybrid | Travel drives H4Δ | Same frvcpy |
| 12 | `fig12_neural_architecture_ablation.png` | Ablation | Cycle-3 fair-LS | 20×3 | History helps | — |
| 13 | `fig13_history_deepsets_paired_effect.png` | Paired Δ | Cycle-3 fair-LS | 20 layouts | Layout-aware SE | Not 60 indep. |
| 14 | `fig14_literature_bks_gap.png` | BKS gaps | fair-LS + INSERTION + n40 | multi | Protocol-separated | Do not mix |
| 15 | `fig15_generalization_scalability.png` | Size transfer | FINAL_n*_fairls | 10–80 | Scales w/ n | n80=3 |
| 16 | `fig16_search_improvement_stages.png` | Search modes | recovered pipe | 20 | Categorical dots | No chain lines |
| 17 | `fig17_quality_vs_compute.png` | Quality/compute | fair_search | 20 | Unequal budgets | Not efficiency proof |
| 18 | `fig18_behavior_cloning_learning_curves.png` | BC curves | training_logs | synth val | Per-arch panels | Sparse steps |
| 19 | `fig19_bc_vs_rl_checkpoint_quality.png` | BC vs RL | reward summary | 20×3 constr. | H4 3/3; H2 1/3 | ≠ hybrid |
| 20 | `fig20_reinforce_reward_dynamics.png` | Rewards | reward_*.csv | RQ4 | Noisy ≠ learning | w/ Fig19 |
| Supp | `supplementary/.../fig21_rl_training_stability.png` | Diagnostics | reward_*.csv | RQ4 | Grad spikes | Correlation only |

## Regenerate

```bash
python scripts/reporting/publication_figures.py
python scripts/reporting/build_results.py --all --skip-decomp
```

## Manuscript sections

- Problem / Method: Figs 01–03  
- Nonlinear charging: Figs 04–11  
- Routing results: Figs 12–17  
- Training / RL: Figs 18–20 (+ supp fig21)  
