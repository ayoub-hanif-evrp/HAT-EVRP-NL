# Search ablation on recovered n=40 constructions

- Objective convention: Full duration
- Units: hours
- Benchmark size: 40
- Instance count: 20
- Methods / checkpoints: History Transformer (H2) / HAT-EVRP-NL (H4) Cycle-3
- Search protocol: ordinary λ=0 starts; matched starts; unequal realized search compute
- Seeds: 42
- Source raw: `results/raw/regression_recovery/RECOVERY_n40_recovered_pipeline.csv`

| Model | Key | Search stage | Stage key | Mean full h | N |
|---|---|---|---|---|---|
| History Transformer Routing (H2) | H2 | Basic Local Search (basic_ls) | basic_ls | 57.1133 | 20 |
| History Transformer Routing (H2) | H2 | Strong Local Search (strong_ls) | strong_ls | 57.8618 | 20 |
| History Transformer Routing (H2) | H2 | Classical Ruin-and-Repair (classical_ruin) | classical_ruin | 57.0343 | 20 |
| History Transformer Routing (H2) | H2 | Neural Position-Aware Repair (neural_insert) | neural_insert | 57.6935 | 20 |
| History Transformer Routing (H2) | H2 | Classical + Neural Insertion Repair (hybrid_insert) | hybrid_insert | 56.8055 | 20 |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | H4 | Basic Local Search (basic_ls) | basic_ls | 57.4126 | 20 |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | H4 | Strong Local Search (strong_ls) | strong_ls | 57.9364 | 20 |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | H4 | Classical Ruin-and-Repair (classical_ruin) | classical_ruin | 56.5603 | 20 |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | H4 | Neural Position-Aware Repair (neural_insert) | neural_insert | 57.7121 | 20 |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | H4 | Classical + Neural Insertion Repair (hybrid_insert) | hybrid_insert | 56.3700 | 20 |

## Interpretation

Classical + Neural Insertion Repair beats historical fair-LS. Not an equal-oracle-budget experiment.
