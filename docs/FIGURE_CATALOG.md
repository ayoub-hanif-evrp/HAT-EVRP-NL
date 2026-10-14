# Figure Catalog — HAT-EVRP-NL (21 main figures)

Contact sheet (inspection only): `results/supplementary/figures/figure_contact_sheet.png`  
Allowlist: `MAIN_FIGURES` in `reporting_paths.py`  
Generator: `scripts/reporting/publication_figures.py`

| # | File | Question | Raw source | Benchmark / N | Protocol | Takeaway | Caveats |
|---|---|---|---|---|---|---|---|
| 01 | `fig01_hat_evrp_nl_architecture.png` | What is HAT-EVRP-NL? | Architecture diagram | — | Method | Neural vs frvcpy vs checker | Schematic |
| 02 | `fig02_montoya_instance_layouts.png` | Benchmark geography | Official XML | n=20 × 3 families | Layout | Spatial diversity tc0/tc1/tc2 | Family labels = Montoya prefixes |
| 03 | `fig03_feasible_ev_routes.png` | Solved routes | `solutions/H2_tc2c40s5cf3_hybrid.json` | n=40 | hybrid_insert | Real routes + charges | One instance |
| 04 | `fig04_nonlinear_charging_curves.png` | NL vs linear curves | XML + surrogates | Curve tech | Physics | Late-SOC slowdown | — |
| 05 | `fig05_charging_time_vs_soc.png` | Marginal time vs SOC | XML PWL slopes | Curve tech | Physics | min/kWh rises with SOC | Segment slopes |
| 06 | `fig06_fixed_route_charging_feasibility.png` | Fixed-route feasibility | `fixed_route_charging_comparison.csv` | 20/size | Fixed seq. | Linear fails true physics | — |
| 07 | `fig07_end_to_end_charging_feasibility.png` | E2E feasibility + route changes | e2e CSVs | 20/size | Frozen H2 | Routing also changes | — |
| 08 | `fig08_soc_charging_schedule.png` | SOC case | schedules JSON | tc0c10s2cf1 | Fixed seq. | EARLY violates Tmax | One route |
| 09 | `fig09_linear_charging_estimation_error.png` | Charge-time error | fixed_route CSV | recorded pairs | Planner hours | EARLY underestimates | Selected by recorded hours |
| 10 | `fig10_station_usage_and_energy.png` | What frvcpy chooses | n40 decomp | 20 | hybrid | Tech visits / energy | Aggregates |
| 11 | `fig11_objective_decomposition.png` | Objective parts | n40 decomp | 20 | hybrid | Travel drives H4 vs H2 | Same frvcpy |
| 12 | `fig12_neural_architecture_ablation.png` | Architecture quality | Cycle-3 fair-LS | 20×3 | fair-LS | History helps | — |
| 13 | `fig13_history_deepsets_paired_effect.png` | Paired Δ | Cycle-3 fair-LS | pairs | fair-LS | H2−H1 / H4−H2 | Seed-correlated pairs |
| 14 | `fig14_literature_bks_gap.png` | BKS gaps | `IMPROVEMENT_bks_gaps_corrected.csv` | 20 | fair-LS Z_lit | ~5% H2 lit gap | n10 only |
| 15 | `fig15_generalization_scalability.png` | Size transfer | FINAL_n*_fairls | 10–80 | fair-LS | Scales with n | n80 = 3 inst. |
| 16 | `fig16_search_improvement_stages.png` | Search stages | recovered pipeline | 20 | multi-mode | Hybrid best mean | Not a forced chain |
| 17 | `fig17_quality_vs_compute.png` | Quality vs compute | fair_search + pipe | 20 | stage-level | Unequal budgets | Not equal-budget |
| 18 | `fig18_behavior_cloning_learning_curves.png` | BC learning | training_logs c3 | synth val | BC | Logged eval points | Sparse steps |
| 19 | `fig19_bc_vs_rl_checkpoint_quality.png` | Controlled BC vs RL | reward_learning summary | 20×3 | construction-only | H4 3/3; H2 1/3 | ≠ Cycle-3 synth-val |
| 20 | `fig20_reinforce_reward_dynamics.png` | Reward dynamics | reward_*.csv | RQ4 RL | REINFORCE | R vs baseline | Window=5 |
| 21 | `fig21_rl_training_stability.png` | Grad/entropy/adv | reward_*.csv | RQ4 RL | Diagnostic | Grad spikes | Correlation ≠ causation |

## Regenerate

```bash
python scripts/reporting/publication_figures.py
python scripts/reporting/build_results.py --all --skip-decomp
```

## Manuscript sections

- Problem / Method: Figs 01–03  
- Nonlinear charging: Figs 04–11  
- Routing results: Figs 12–17  
- Training / RL: Figs 18–21  
