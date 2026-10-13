# Search ablation on recovered n=40 constructions

- Objective convention: Full duration
- Units: hours
- Benchmark size: 40
- Instance count: 20
- Methods / checkpoints: H2/H4 Cycle-3
- Search protocol: ordinary λ=0 starts; matched starts; unequal realized search compute
- Seeds: 42
- Source raw: `results/raw/regression_recovery/RECOVERY_n40_recovered_pipeline.csv`

| Method | Stage | Mean full h | N |
|---|---|---|---|
| H2 | basic_ls | 57.1133 | 20 |
| H2 | strong_ls | 57.8618 | 20 |
| H2 | classical_ruin | 57.0343 | 20 |
| H2 | neural_insert | 57.6935 | 20 |
| H2 | hybrid_insert | 56.8055 | 20 |
| H4 | basic_ls | 57.4126 | 20 |
| H4 | strong_ls | 57.9364 | 20 |
| H4 | classical_ruin | 56.5603 | 20 |
| H4 | neural_insert | 57.7121 | 20 |
| H4 | hybrid_insert | 56.3700 | 20 |

## Interpretation

Hybrid_insert beats historical fair-LS. Not an equal-oracle-budget experiment.
