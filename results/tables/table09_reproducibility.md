# Reproducibility checklist for strongest n40 result

- Objective convention: Full duration
- Units: hours
- Benchmark size: 40
- Instance count: 20
- Methods / checkpoints: History Transformer + DeepSets + hybrid insert (best mean)
- Search protocol: ordinary λ=0, 3 samples, strong LS, classical ruin 6/600 + neural insert 6
- Seeds: 42
- Source raw: `scripts/experiments/run_regression_recovery.py`

| Item | Value |
|---|---|
| Checkpoint | checkpoints/H4_n10_seed42_c3.pt (internal key H4) |
| Construction | decode sample=(i>0), cost_lambda=0, no force_first |
| Samples | 3 |
| LS | strong, passes=3, budget=4000 |
| Hybrid | ruin iters=6 budget=600 + neural_insert iters=6 repair_insertion=True |
| Command | python scripts/experiments/run_regression_recovery.py --stage pipeline |
| Mean objective | 56.3700 h (see RECOVERY pipeline) |
| Reporting | python scripts/reporting/build_results.py --all |

## Interpretation

Preserve best incumbent; never accept a worse repair. Single seed — exploratory. Checkpoint filenames retain internal H4 key for reproducibility.
