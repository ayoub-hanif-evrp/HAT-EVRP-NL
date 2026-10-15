# Search ablation on recovered n=40 constructions

- Objective convention: Full duration
- Units: hours
- Benchmark size: 40
- Instance count: 20
- Methods / checkpoints: History Transformer / History Transformer + DeepSets (Cycle-3)
- Search protocol: ordinary λ=0 starts; matched starts; unequal realized search compute
- Seeds: 42
- Source raw: `results/raw/regression_recovery/RECOVERY_n40_recovered_pipeline.csv`

| Method | Search mode | Mean full h | N |
|---|---|---|---|
| History Transformer | Basic local search | 57.1133 | 20 |
| History Transformer | Strong local search | 57.8618 | 20 |
| History Transformer | Classical ruin–repair | 57.0343 | 20 |
| History Transformer | Neural insert repair | 57.6935 | 20 |
| History Transformer | Hybrid insert repair | 56.8055 | 20 |
| History Transformer + DeepSets | Basic local search | 57.4126 | 20 |
| History Transformer + DeepSets | Strong local search | 57.9364 | 20 |
| History Transformer + DeepSets | Classical ruin–repair | 56.5603 | 20 |
| History Transformer + DeepSets | Neural insert repair | 57.7121 | 20 |
| History Transformer + DeepSets | Hybrid insert repair | 56.3700 | 20 |

## Interpretation

Hybrid insert repair has the best mean among reported modes. Not an equal-oracle-budget experiment.
