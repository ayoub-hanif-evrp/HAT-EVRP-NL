# Publication Quality Report — HAT-EVRP-NL

**Date:** 2026-10-15  
**Base reviewed commit:** `6d47ff6cadeb47899d960a2b4c214ef923f4c71c`  
**Display naming:** centralized in `presentation_names.py` (`METHOD_DISPLAY` / `METHOD_DISPLAY_NAMES`)

---

## 1. Results quality (evidence strength)

| Research question | Verdict | Evidence |
|---|---|---|
| RQ1 History vs State-only | **Moderate** | Mean fair-LS n=10: 19.813 vs 19.993 h. **Seed-sensitive:** ≈ −0.578 / +0.040 / ≈0 h on seeds 42–44 (Fig. 12B). Average ≠ consistent across trained models. |
| RQ2 DeepSets vs History | **Inconclusive / weak** | Mean 19.886 vs 19.813 h; paired Hist+DS − History straddles zero (Fig. 13). n=40 hybrid single-seed Δ ≈ −0.436 h is exploratory. Seed-42 construction-only: identical objectives on all 20 instances historically noted. |
| RQ3 Nonlinear vs linear charging | **Strong** | Fixed-route + E2E feasibility (Figs 6–7); true-physics failures distinct from planner rejection. |
| RQ4 BC vs RL | **Protocol-dependent** | Official n=10 construction: Hist+DS 3/3 RL better; History 1/3. Cycle-3 synth-val endpoints often worse; checkpoints BC-selected. Not hybrid-search RL. |
| BKS / scale | **Honestly limited** | Fair-LS ~5% lit gap; expensive hybrid ~0.03%; n=40 ~17%. n=80 = 3 instances (exploratory Fig. 15). |

**Newly executed this pass:** figure/table regeneration and naming/visual QA only.  
**Not newly trained:** no new GPU training runs; decisive matched ablations (prefix-only history, pooling, open-route DeepSets, equal-budget classical, NL-oracle factorial, hybrid-pipeline RL) remain **queued / incomplete** — configs exist historically where available; do not invent results.

---

## 2. Figure quality report

All 20 main figures regenerated at ≥300 dpi; contact sheet rebuilt. Visual inspection of full-resolution PNGs completed (see `docs/FIGURE_QUALITY_REVIEW.md`).

| File | Purpose | Changes | Visual QA | Data-supported |
|---|---|---|---|---|
| fig01 | Architecture | Arrow routing; no H-codes; learned vs algorithmic | Pass | Schematic |
| fig06–07 | Charging feasibility | Per-bar model names; three outcome classes | Pass | Yes |
| fig12 | Architecture ablation | Descriptive names + **per-seed companion** | Pass | Yes |
| fig13 | Paired effects | Descriptive pair labels; layout units | Pass | Yes |
| fig14 | BKS gaps | Protocol panels; near-zero annotation | Pass | Yes |
| fig15 | Size transfer | Exploratory; absolute + Δ + feas | Pass | Yes (thin n80) |
| fig16 | Search modes | Grouped bars; no legend overlap | Pass | Yes |
| fig17 | Quality vs compute | Exploratory unequal-budget title | Pass | Yes |
| fig18–20 | Learning / RL | Descriptive names; construction-only caveats | Pass | Yes |

---

## 3. Table quality report

Main tables regenerated with descriptive Method columns (no bare H-codes in display cells). Protocol metadata retained. Key tables: `table03_main_models.md`, `table02`, `table04`, `table06_h2_vs_h4.md` (renamed content), `table08`, `benchmark_protocol_comparison.md`, learning tables 10–14, `table06_bc_vs_rl.md`.

Raw CSVs retain internal keys for reproducibility.

---

## 4. Final statistical report (defensible)

- **Primary n=10 fair-LS (3 seeds × 20):** History Transformer strongest neural mean; DeepSets not a consistent win.  
- **Seed variability:** History − State-only is **not** stable across seeds 42–44.  
- **Layout-aware pairing:** use n=20 layouts (seed-averaged), not 60 pseudo-independent rows (`scripts/analysis/paired_stats.py` + Fig. 13).  
- **n=40 hybrid:** History+DeepSets 56.370 vs History 56.806 h (20/20 feasible); single seed; wins/ties/losses 7/4/9.  
- **Feasibility:** Cycle-3 fair-LS rows fully feasible where reported; nonlinear study shows large true-physics gaps for linear planners.

---

## 5. Reproducibility report

```bash
python scripts/reporting/publication_figures.py
python -c "… build_tables / write_learning_tables / write_bc_rl_table …"
python -m compileall -q .
python -m unittest discover -s tests -p 'test_*.py' -v   # 73 tests OK
```

Sanity eval (run separately if needed): `python evaluate.py --mode sanity --data dataset`

Generated: `results/figures/fig01`–`fig20`, `results/supplementary/figures/fig21*`, contact sheet, refreshed `results/tables/*`, updated `paper_draft.md`, `docs/FIGURE_CATALOG.md`, this report.

**Unresolved / not claimed finished:** expensive new training; equal-budget classical vs neural; prefix-only / pooling / open-route DeepSets matched studies; RL inside final hybrid pipeline; GitHub Actions not re-verified in this session.

---

## 6. Publication-readiness assessment

| Area | Ready? |
|---|---|
| Descriptive naming in figures/tables/paper captions | **Yes** |
| Visual figure quality (main set) | **Yes** (restrained academic style) |
| Scientific honesty of claims | **Yes** |
| Complete decisive experiment suite | **No** — several matched studies still outstanding |
| SOTA / DeepSets / equal-budget efficiency claims | **Correctly not made** |

**Bottom line:** Presentation layer is publication-ready for an honest conference paper. Strongest scientific story remains nonlinear charging + cautious History Transformer gains with explicit seed variability. Full architecture (History + DeepSets) should not be oversold.
