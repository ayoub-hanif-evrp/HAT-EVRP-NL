# Figure Audit — HAT-EVRP-NL

Baseline inventory before cleanup: **25 PNGs** in `results/figures/`.  
After cleanup: **8 main-paper PNGs**; supporting plots in `results/supplementary/figures/`; superseded plots in `results/archive/figures_legacy/`.

Single source of truth: `scripts/reporting/publication_figures.py` + allowlist `MAIN_FIGURES` in `reporting_paths.py`.

---

## Final main-paper set (8)

| # | File | Question answered | Raw source |
|---|---|---|---|
| 1 | `fig01_architecture.png` | What does HAT-EVRP-NL do? | Architecture diagram (neural vs algorithmic) |
| 2 | `fig02_nonlinear_charging_curves.png` | How does NL charging differ from linear? | Official XML curves + surrogates |
| 3 | `fig03_charging_model_feasibility.png` | What happens to feasibility under linear assumptions? | `raw/nonlinear_charging/*` |
| 4 | `fig04_charging_schedule_case.png` | What SOC/charging decisions explain the gap? | Saved schedules + true-curve replay |
| 5 | `fig05_objective_decomposition.png` | What contributes to the objective? | `RECOVERY_n40_objective_decomposition.csv` |
| 6 | `fig06_architecture_ablation.png` | Do history / DeepSets improve quality? | Cycle-3 n10 fair-LS seed CSVs |
| 7 | `fig07_learning_and_reward.png` | What effect do BC and RL have? | Cycle-3 logs + `raw/reward_learning/` |
| 8 | `fig08_quality_compute.png` | Quality vs computational effort? | `RECOVERY_n40_fair_search.csv` + pipeline |

---

## Disposition of former figures

| Old file | Decision | Rationale | Replacement / location |
|---|---|---|---|
| `fig01_architecture.png` (old) | Redesign | Sparse / oversized boxes | New `fig01_architecture.png` |
| `fig02_problem_example.png` | Supplementary | Useful context, not a core RQ figure | `supplementary/figures/` |
| `fig03_charging_curves.png` | Replace | Superseded by clearer NL vs linear | Main `fig02_*` |
| `fig04_objective_decomposition_n40.png` | Supplementary | Same n40 hybrid decomp; protocol twin of main fig05 | `supplementary/figures/` |
| `fig05_soc_charging_trace.png` | Supplementary | Single-route SOC; case study now fig04 | `supplementary/figures/` |
| `fig06_main_benchmark_n10_fairls.png` | Archive | Duplicate of architecture ablation | Main `fig06_*` |
| `fig07_scaling_fairls.png` | Supplementary | Scaling, not one of the 8 core questions | `supplementary/figures/` |
| `fig08_search_ablation_n40_recovered.png` | Replace | Quality-only bars ≠ efficiency | Main `fig08_quality_compute.png` |
| `fig09_h2_vs_h4_n40_hybrid.png` | Supplementary | Instance-level H2/H4; unequal compute | `supplementary/figures/` |
| `fig10_charging_ablation_feasibility.png` | Supplementary | Direct CHARGE vs frvcpy; supporting | `supplementary/figures/` |
| `fig11_training_supervision_flow.png` | Supplementary | Overlaps architecture/training path | `supplementary/figures/` |
| `fig12_learning_curves_h1h4.png` | Supplementary | Detailed BC/RL curves; summarized in fig07 | `supplementary/figures/` |
| `fig12b_bc_and_rl_losses_separate.png` | Supplementary | Loss ≠ reward; supporting | `supplementary/figures/` |
| `fig13_rl_effect_h2_h4.png` | Supplementary | Historical Cycle-3 synth-val RL | `supplementary/figures/` |
| `fig14_reward_dynamics_unavailable.png` | Archive | Placeholder, not scientific | `archive/figures_legacy/` |
| `fig15_bc_vs_rl_ablation.png` | Supplementary | Historical aggregate; controlled RQ4 in fig07 | `supplementary/figures/` |
| `fig16_hierarchical_policy.png` | Supplementary | Policy schematic | `supplementary/figures/` |
| `fig17_history_deepsets_contribution.png` | Archive | Duplicate of fig06 | `archive/` / superseded by `fig06_*` |
| `fig_main02_nonlinear_vs_linear_curves.png` | Redesign→main fig02 | Renamed/cleaned | `fig02_nonlinear_charging_curves.png` |
| `fig_main03_*.png` + `fig_main03c_*` | Merge | Same three models, two protocols | `fig03_charging_model_feasibility.png` |
| `fig_main03b_paired_charging_time_deltas.png` | Supplementary (revised) | Selection bias if pooled | `supp_paired_charging_time_by_size.png` |
| `fig_main03d_schedule_comparison_example.png` | Replace | Text dump, not a plot | `fig04_charging_schedule_case.png` + Markdown table |
| `fig_main04_objective_decomposition_nl.png` | Archive | Different population than n40 hybrid | Archived; main uses recovered n40 |
| `fig_main07_reward_dynamics.png` | Redesign | Jagged + flat 100% feasibility panel | Panel C of `fig07_*` (smoothed) |
| `fig_main07b_bc_vs_rl_official.png` | Merge into fig07 | Paired BC/RL is panel B | `fig07_learning_and_reward.png` |

---

## Regeneration contract

```bash
python scripts/reporting/publication_figures.py
# or
python scripts/reporting/build_results.py --all --skip-decomp
```

- Legacy builders in `build_results.py` / `learning_presentation.py` write **only** to `results/supplementary/figures/`.
- `charging_and_reward_figures.py` writes **tables only**.
- `enforce_figure_layout()` ensures `results/figures/` contains exactly `MAIN_FIGURES`.
- Tests: `tests/test_publication_figures.py`.

---

## Omitted for lack of evidence

- Continuous search trajectories / anytime curves (only aggregate end-state runtime/oracle available).
- Equal-budget wall-clock superiority plots (not established).
- Fabricated reward histories for Cycle-3 (logs lack sampled rewards).
