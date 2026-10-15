# Phase 1 Change Report — Correctness & Reproducibility

## Summary

Implemented verified correctness fixes for training CSV logging, route validation,
wall-clock classical evaluation bookkeeping, route fingerprints, local-search
operator attribution, and independent objective decomposition. Added regression
tests, methodology audit, experiment protocol, paired-stats helper, and a fast CI
workflow. **No historical checkpoints, XML data, or raw experimental CSVs were
overwritten. No expensive multi-seed experiments were launched.**

## Files changed

| File | Why |
|---|---|
| `train.py` | `write_training_csv()` unions BC/RL fieldnames; fixes `ValueError` on combined logs |
| `charging.py` | Nonfinite charge/duration rejection; stored-vs-simulated duration check; independent travel in `decompose_route` + `identity_ok`; `route_sequence_fingerprint()` |
| `evaluate.py` | `_classical_until_time` no longer runs extra greedy after the budget; builds `BaselineResult` directly; tracks LS wall time; soft remaining-time budget shrink |
| `solver_ls.py` | Accepted improvements credited to the operator that created the incumbent |
| `scripts/experiments/run_nonlinear_charging_study.py` | Uses invariant route fingerprints |
| `tests/test_phase1_correctness.py` | New regression tests |
| `scripts/analysis/paired_stats.py` | Layout-aware paired Δ + descriptive CI |
| `docs/METHODOLOGY_AUDIT.md` | RQ1–RQ4 strengths/confounders/plan |
| `docs/EXPERIMENT_PROTOCOL.md` | Commands, metrics, compute caution |
| `.github/workflows/ci-fast.yml` | Fast CI for correctness tests |
| `docs/PHASE1_CHANGE_REPORT.md` | This report |

## Tests

Run locally:

```bash
python -m unittest tests.test_phase1_correctness -v
python -m unittest tests.test_repair_and_validation tests.test_reporting_pipeline -q
```

## Not done (explicitly deferred)

- Prefix-only history ablation training (RQ1).
- DeepSets pooling/open-only multi-seed ablations (RQ2).
- Re-running the full nonlinear e2e study to refresh decision-change rates (RQ3) — code ready; regenerate when authorized.
- Reward rescaling / new RL campaigns (RQ4).
- Push/merge (requires approval).

## Scientific claim status (unchanged numerically)

See `results/SCIENTIFIC_RESULTS_AUDIT.md` and `docs/METHODOLOGY_AUDIT.md`.
No claim was strengthened without new evidence.
