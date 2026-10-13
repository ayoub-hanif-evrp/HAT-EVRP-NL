# Results index (publication layer)

Presentation (`figures/`, `tables/`) is separated from evidence (`raw/`, `archive/`).
Manuscript: [`../paper_draft.md`](../paper_draft.md).

## Research questions — key evidence

| RQ | Question | Primary evidence |
|---|---|---|
| RQ1 | History vs state-only | `table03_main_models.md` — H2 19.813 vs H1 19.993 h (n10 fair-LS) |
| RQ2 | DeepSets vs history | Same table — H4 19.886 vs H2 19.813; n40 hybrid inconclusive |
| RQ3 | Nonlinear vs linear charging | [`NONLINEAR_CHARGING_STUDY.md`](NONLINEAR_CHARGING_STUDY.md), `table15_*` |
| RQ4 | BC vs RL | [`REWARD_AND_LEARNING_STUDY.md`](REWARD_AND_LEARNING_STUDY.md), `table06_bc_vs_rl.md` |

## Current strongest supported routing results

| Setting | Configuration | Mean full h | Notes |
|---|---|---:|---|
| n=40 | H4 Cycle-3 + ordinary λ=0 + hybrid_insert | **56.370** | Single seed; unequal search compute |
| n=40 | H2 Cycle-3 + ordinary λ=0 + hybrid_insert | **56.806** | Same protocol as H4 row |
| n=10 | H2 Cycle-3 fair-LS | **19.813** | Multi-seed Cycle-3 baseline |

## Nonlinear charging (RQ3) — executed full coverage

Fixed-route plans feasible under **true** Montoya physics:

| n | TRUE-NL | LINEAR-SECANT | LINEAR-EARLY-RATE |
|---:|---:|---:|---:|
| 10 | 20/20 | 16/20 | 16/20 |
| 20 | 20/20 | 12/20 | 12/20 |
| 40 | 20/20 | 11/20 | 11/20 |

## Learning (RQ4)

- Historical Cycle-3 synth-val: RL improves best-BC on 1/3 H2, 0/3 H4 seeds.
- Controlled RQ4 construction-only official n10: H4 RL better 3/3; H2 1/3.
- Do not mix these protocols.

## Main publication figures (prefer these)

| Fig | File |
|---|---|
| Architecture / training | `fig11_training_supervision_flow.png`, `fig01_architecture.png` |
| NL vs linear curves | `fig_main02_nonlinear_vs_linear_curves.png` |
| True-physics feasibility | `fig_main03_linear_plan_true_feasibility.png`, `fig_main03c_e2e_true_feasibility.png` |
| Objective decomposition | `fig_main04_objective_decomposition_nl.png`, `fig04_objective_decomposition_n40.png` |
| Matched neural quality | `fig06_main_benchmark_n10_fairls.png`, `fig17_history_deepsets_contribution.png` |
| Learning curves | `fig12_learning_curves_h1h4.png`, `fig12b_bc_and_rl_losses_separate.png` |
| Reward dynamics (real logs) | `fig_main07_reward_dynamics.png`, `fig_main07b_bc_vs_rl_official.png` |
| Quality vs compute | `fig08_search_ablation_n40_recovered.png` |

Secondary / legacy plots remain in `figures/` or `archive/`. **Do not** cite `fig14_reward_dynamics_unavailable.png` (archived placeholder).

## Main tables

- `table01`–`table03` — benchmark, objectives, main models
- `table15_nonlinear_vs_linear_charging.md` — **RQ3**
- `table06_bc_vs_rl.md` / `table12_bc_vs_rl.md` — **RQ4**
- `table08_bks_comparison.md` — literature \(Z_\mathrm{lit}\) gaps
- Architecture/search ablations: `table04`, `table06_h2_vs_h4`, `table14`

## Raw evidence

- `raw/nonlinear_charging/` — fixed-route + e2e charging study
- `raw/reward_learning/` — controlled BC/RL + reward CSVs
- `raw/cycle3/` — Cycle-3 fair-LS
- `raw/regression_recovery/` — n40 recovery
- `raw/training_logs/` — Cycle-3 training CSVs + derived aggregates

## Key reports

- [`NONLINEAR_CHARGING_STUDY.md`](NONLINEAR_CHARGING_STUDY.md)
- [`REWARD_AND_LEARNING_STUDY.md`](REWARD_AND_LEARNING_STUDY.md)
- [`FINAL_REPORT.md`](FINAL_REPORT.md)
- [`LEARNING_RESULTS_REPORT.md`](LEARNING_RESULTS_REPORT.md)
- [`../docs/TRAINING_AND_REWARD_METHODOLOGY.md`](../docs/TRAINING_AND_REWARD_METHODOLOGY.md)

## Regenerate

```bash
python scripts/reporting/charging_and_reward_figures.py
python scripts/reporting/build_results.py --all --skip-decomp
```
