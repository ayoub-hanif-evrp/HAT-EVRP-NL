# Objective decomposition with nonlinear charging (n=40 hybrid)

- Objective convention: FULL=travel+service+charge; LIT=travel+charge
- Units: hours
- Benchmark size: 40
- Instance count: 20
- Methods / checkpoints: History Transformer / History + DeepSets (Cycle-3 recovered hybrid)
- Search protocol: ordinary λ=0 + strong LS + hybrid_insert
- Seeds: 42
- Source raw: `results/raw/regression_recovery/RECOVERY_n40_objective_decomposition.csv`

| Method | Travel h | Service h | Nonlinear charge h | Full h | Lit h | Mean charge visits | Charge share |
|---|---|---|---|---|---|---|---|
| History Transformer | 33.7454 | 20.0000 | 3.0601 | 56.8055 | 36.8055 | 9.85 | 4.9% |
| History Transformer + DeepSets | 33.2189 | 20.0000 | 3.1511 | 56.3700 | 36.3700 | 9.65 | 5.2% |

## Interpretation

Charging hours come from exact F_s(q_out)-F_s(q_in) on validated RoutePlan.stops. frvcpy optimizes the fixed-route charging subproblem only. History + DeepSets’ better full duration is driven mainly by lower travel despite slightly higher mean charging.
