# Figure Quality Review (publication pass)

Date: 2026-10-15  
Generator: `scripts/reporting/publication_figures.py`  
Display names: `presentation_names.py` (`METHOD_DISPLAY_NAMES`)

## Naming

All main-paper figure legends, titles, and axis labels use descriptive method names
(State-only Pointer, History Transformer, DeepSets Fleet Model, History Transformer + DeepSets).
Internal H0–H4 keys remain only in raw CSVs and checkpoint filenames.

## Visual inspection (full-resolution PNG)

| Fig | Status | Notes |
|---|---|---|
| 01 | Pass | Sections A/B/C; arrows routed around B title; no gradient-through-frvcpy implication |
| 02–05 | Pass | Geography, curves, marginal SOC — clean |
| 06–07 | Pass | Per-bar charging-model names; true-feas / planner-OK-fail / planner-reject separated; counts labeled |
| 08–09 | Pass | SOC schedule + estimation error |
| 10–11 | Pass | Descriptive architecture names; travel dominates Δ |
| 12 | Pass | Two panels: aggregate seed means ± std + **per-seed** History−State-only / Hist+DS−History |
| 13 | Pass | Layout-aware paired units (n=20); zero line; descriptive pair labels |
| 14 | Pass | Three protocol panels with independent scales; near-zero hybrid annotated |
| 15 | Pass | Exploratory title; absolute + Δ + feasibility; n=80 flagged |
| 16 | Pass | Grouped bars; legend clear of data; categorical modes |
| 17 | Pass | Exploratory unequal-budget title; legend outside |
| 18–20 | Pass | Descriptive panel titles; BC vs RL construction-only caveats |
| 21 (supp) | Pass | Diagnostics only |

Contact sheet: `results/supplementary/figures/figure_contact_sheet.png`

## Claims alignment

Figures do **not** claim: DeepSets superiority, equal-budget search efficiency, SOTA BKS, or reliable RL for all architectures.
