# Final Acceptance Report — HAT-EVRP-NL

**Base commit reviewed:** `5680dfbacd2d6e1960120eda263fb0c699f61e94`  
**Date:** 2026-10-15

---

## A. Figures

| Fig | Change | Visual inspection |
|---|---|---|
| 01 | Redesigned spacing; B title to the right of the sequences→frvcpy arrow corridor; C title below B boxes with clear gap | Pass (arrow corridor x≈2.3; B title x≥4.3) |
| 06 | Faceted panels per size (n=10/20/40); model labels under bars; three outcome classes preserved | Pass |
| 07 | Faceted A panels + B route-change panel; removes crowded single-axis labels | Pass |
| 15 | Annotation spacing + exploratory n=80 disclaimer | Pass |
| 16 | Legend above axes (outside data region) | Pass |
| 02–05, 08–14, 17–20, 21 | Regenerated; descriptive names retained | Pass |

Contact sheet rebuilt. Display names remain centralized in `presentation_names.py`.

---

## B. Tables

- Figure-count header in `paper_draft.md` corrected to **20 main + 1 supplementary**.
- Fig 6/7 captions updated for facet layout.
- No experimental numbers changed.
- Protocol-separated benchmark table already uses descriptive names.

---

## C. Tests

```text
python -m compileall -q .
python -m unittest discover -s tests -p 'test_*.py' -v
python evaluate.py --mode sanity --data dataset
python scripts/reporting/publication_figures.py
```

(Run results recorded in this session: compileall OK; unittest suite previously 73 OK — re-run below; sanity OK; figures regenerated OK.)

GitHub Actions: **not re-confirmed** in this session.

---

## D. Scientific evidence

**Executed now (lightweight, matched where possible):**
- `run_lightweight_architecture_compare.py` → construction-only n=10, seed 42:
  - State-only 22.4990 · History 21.8444 · DeepSets 22.6678 · History+DeepSets **21.8444** · Open-route DeepSets 21.8001 (unmatched ckpt).
  - Confirms H2≡H4 on this seed under construction-only; H4_open not controlled.

**Still deferred (need training / authorization):**
- Prefix-only history ablation (~40–80 CPU-h)
- Mean vs sum pooling + ≥5 seeds (~80–160 CPU-h)
- Oracle×feature nonlinear factorial (~20–40 CPU-h)
- RL under hybrid_insert (~10–20 CPU-h)
- Equal wall-clock classical vs neural

**Supported conclusions (unchanged):** History modest but seed-sensitive; DeepSets not consistent; nonlinear charging strong; RL protocol-dependent; n=40 ~17% BKS gap.

---

## E. Remaining limitations

- Prefix-history confound not experimentally resolved.
- DeepSets pooling / open-route lack matched multi-seed Cycle-3 evidence.
- Equal-budget search efficiency not established.
- n=80 exploratory (3 instances).
- No CI confirmation this pass.

---

## F. Final verdict

| Dimension | Status |
|---|---|
| Presentation-ready | **Yes** (figures/tables/names) |
| Code/test-ready | **Yes** (local suite; re-verify below) |
| Reproducible | **Yes** for reported historical + lightweight P0 |
| Scientifically validated (decisive ablations) | **Partial** — core claims honest; several decisive experiments still deferred |
| Publication-ready | **Yes for an honest conference paper** with explicit limitations; **not** for overstated DeepSets / SOTA / equal-budget claims |
