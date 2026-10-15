# Figure cleanup acceptance — no bottom commentary

Date: 2026-10-15  
Baseline reviewed commit: `8c179f3413b299d68d9f4ccc1b7122f0302f8beb`  
Scope: final figure corrections, caption relocation, regeneration, visual QA.

## 1. Figures changed (this pass)

| Figure | File | Change |
|---|---|---|
| 1 | `fig01_hat_evrp_nl_architecture.png` | Removed ablation/policy bottom notes; kept color key only |
| 7 | `fig07_end_to_end_charging_feasibility.png` | Wider gap + right-side `% changed routes` ylabel; no footer |
| 12 | `fig12_neural_architecture_ablation.png` | Extra y-margin so −0.58 label is not clipped |
| 14 | `fig14_literature_bks_gap.png` | Removed in-plot “≈0% / scale ≠ A” note |
| 15 | `fig15_generalization_scalability.png` | Unclipped Δ labels; near-zero n=80 shown by marker on baseline; caveats in caption |
| 16 | `fig16_search_improvement_stages.png` | Truncated bars → categorical dots; axis from 0; legend outside |
| 17 | `fig17_quality_vs_compute.png` | Title cleaned (unequal-budget note → caption) |
| 21 | `fig21_rl_training_stability.png` | Regenerated; no bottom commentary |
| All 1–20 | `results/figures/*.png` | Regenerated; all `fig.text` / `plt.figtext` footers removed from generators |

## 2. Bottom explanatory comments removed

Confirmed in plotting scripts (`publication_figures.py`, `learning_presentation.py`, `charging_and_reward_figures.py`, `build_results.py`):

- No `fig.text(...)` / `plt.figtext(...)` calls remain.
- Test: `test_no_bottom_fig_text_in_plot_scripts` — **PASS**.
- Visual inspection of main figures 1–20 and supplementary fig21: **no bottom footnotes / finding summaries**.

## 3. Captions preserve scientific caveats

Updated / verified in `paper_draft.md`:

- Fig 1: ablations + policy-gradient / post-decode charging.
- Fig 7: frozen History weights; end-to-end sequence changes.
- Fig 14: panel B near-zero gap uses a different vertical scale.
- Fig 15: n=80 has only 3 instances; not a strong scalability claim.
- Fig 16: categorical alternatives; not sequential; not equal-budget.
- Fig 17: unequal realized budgets (exploratory).

## 4. Visual inspection summary

| Fig | Verdict |
|---|---|
| 01 | Clean architecture; color key only |
| 02–05 | OK (layouts, routes, charging curves, marginal time) |
| 06 | Faceted feasibility; no footer |
| 07 | Spacing fixed; right ylabel clear of n=40 panel |
| 08–11 | OK (SOC schedule, error, station usage, decomposition) |
| 12 | Seed-sensitive paired bars; −0.58 readable |
| 13 | Layout-aware paired effects OK |
| 14 | Protocol-separated BKS; no in-plot scale disclaimer |
| 15 | +0.00 fully inside panel B |
| 16 | Dot plot; full axis from 0; values match CSV (56–60 h) |
| 17 | Legend outside; no unequal-budget footer |
| 18–20 | Learning panels OK |
| 21 | Supplementary diagnostics OK |

## 5. Verification commands

```text
python -m compileall -q .
python -m unittest discover -s tests -p 'test_*.py' -v
python evaluate.py --mode sanity --data dataset
python scripts/reporting/publication_figures.py
python scripts/reporting/build_results.py --all --skip-decomp
```

Results:

- **compileall**: OK
- **unittest**: 78 tests, **0 failures** (ResourceWarnings only)
- **sanity**: passed (`tc0c10s2cf1`)
- **publication_figures**: 20 main + fig21 OK
- **build_results**: tables regenerated; raw CSVs not overwritten with fabricated values

## 6. Numerical integrity

- Raw CSV means unchanged (e.g. n=40 hybrid_insert H2 ≈ 56.81 h, H4 ≈ 56.37 h).
- Fig16 plotted means match `RECOVERY_n40_recovered_pipeline.csv`.
- Fig15 Δ values derived from `FINAL_n*_fairls.csv` (n=80 Δ = +0.00).
- No historical experimental results overwritten.

## 7. Remaining scientific limitations (unchanged)

Visual polish does **not** complete scientific validation.

1. History Transformer gains remain modest and seed-sensitive.
2. DeepSets benefit remains inconclusive.
3. Matched multi-seed DeepSets pooling / open-route ablations still needed.
4. Prefix-only route-history ablation still needed.
5. Equal-budget classical-versus-neural comparison still needed.
6. Nonlinear oracle-versus-feature factorial study still needed.
7. RL evaluation inside the final hybrid pipeline still needed.
8. n=80 size transfer uses only 3 instances.

Do not invent unexecuted results. Do not push without approval.
