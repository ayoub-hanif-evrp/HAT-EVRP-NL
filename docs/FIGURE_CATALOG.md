# Figure Catalog — HAT-EVRP-NL

Contact sheet: `results/supplementary/figures/figure_contact_sheet.png`  
Allowlist: `MAIN_FIGURES` in `reporting_paths.py` (20 main PNGs)  
Supplementary: `SUPPLEMENTARY_FIGURES` (fig21 diagnostics)  
Generator: `scripts/reporting/publication_figures.py`  
Display names: `presentation_names.py` (no H0–H4 in reader-facing legends)  
Audit: `results/SCIENTIFIC_RESULTS_AUDIT.md` · Quality: `docs/FIGURE_QUALITY_REVIEW.md`

| # | File | Question | Raw source | N / protocol | Takeaway | Caveats |
|---|---|---|---|---|---|---|
| 01 | `fig01_hat_evrp_nl_architecture.png` | Architecture | Schematic | — | Neural vs frvcpy vs checker | Schematic |
| 02 | `fig02_montoya_instance_layouts.png` | Geography | Official XML | n=20×3 | Spatial diversity | Family prefixes |
| 03 | `fig03_feasible_ev_routes.png` | Solved routes | solutions JSON | n=40 hybrid | Real routes+charges | One instance |
| 04 | `fig04_nonlinear_charging_curves.png` | NL curves | XML breakpoints | techs | Late-SOC slowdown | — |
| 05 | `fig05_charging_time_vs_soc.png` | Marginal time | PWL slopes | techs | min/kWh rises w/ SOC | — |
| 06 | `fig06_fixed_route_charging_feasibility.png` | Fixed-route feas. | fixed_route CSV | 20/size | Linear fails physics | plan_* rows |
| 07 | `fig07_end_to_end_charging_feasibility.png` | E2E feas. | e2e CSVs | 20/size | Routing changes too | Frozen History Trans. |
| 08 | `fig08_soc_charging_schedule.png` | SOC case | schedules JSON | one route | EARLY vs Tmax | Case study |
| 09 | `fig09_linear_charging_estimation_error.png` | Time error | fixed_route CSV | recorded | EARLY bias | Selected subset |
| 10 | `fig10_station_usage_and_energy.png` | frvcpy choices | n40 decomp | 20 hybrid | Visits/energy/time | 3 panels |
| 11 | `fig11_objective_decomposition.png` | Obj. parts | n40 decomp | 20 hybrid | Travel drives Δ | Same frvcpy |
| 12 | `fig12_neural_architecture_ablation.png` | BC and RL seed means | bc_rl_full_v1 P0 n=10 | 20×5 | Overlapping seeds | Separate BC and RL |
| 13 | `fig13_history_deepsets_paired_effect.png` | Paired RL−BC | bc_rl_full_v1 P0 n=10 | 5 seeds | RT mean only negative | One point per seed |
| 14 | `fig14_literature_bks_gap.png` | BKS gaps | bc_rl_full_v1 n=10 | 20×5 | P0 ~21–25%, P2 ~5–8% | Protocols stay separate |
| 15 | `fig15_generalization_scalability.png` | Size transfer | bc_rl_full_v1 P0 | 20 at n=80 | n=10 policy | RL n=80 RT+DS worse |
| 16 | `fig16_search_improvement_stages.png` | Categorical protocols | bc_rl_full_v1 n=10 | 20×5 | P0/P1/P2 | Unequal budgets |
| 17 | `fig17_quality_vs_compute.png` | Quality/oracle calls | bc_rl_full_v1 n=10 | 20×5 | Unequal budgets | ~55 vs ~160 calls |
| 18 | `fig18_behavior_cloning_learning_curves.png` | BC curves | training.csv | 100 val | Five seeds | Star = best log |
| 19 | `fig19_bc_vs_rl_checkpoint_quality.png` | BC vs RL | eval P0 n=10 | 20×5 | RT 3/5; others 1/5 | Construction only |
| 20 | `fig20_reinforce_reward_dynamics.png` | Normalized reward | reward.csv | 300 steps | Noisy traces | With Fig 19 |
| Supp | `supplementary/.../fig21_rl_training_stability.png` | Diagnostics | reward_*.csv | RQ4 | Grad spikes | Correlation only |

**Naming:** Reader-facing labels use descriptive names from `presentation_names.py` (no bare H1–H4).

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
