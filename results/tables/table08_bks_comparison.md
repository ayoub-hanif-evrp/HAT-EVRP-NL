# Literature-compatible BKS gaps (corrected service convention)

- Objective convention: Lit gap uses (full − Σ service) vs Froger Table 13 travel+charging BKS
- Units: % gap
- Benchmark size: n=10
- Instance count: 20 official instances per method (5 method-summary rows — not 5 instances)
- Methods / checkpoints: Greedy+LS through History+DeepSets (Cycle-3 fair-LS)
- Search protocol: fair-LS Cycle-3; service-time correction applied
- Seeds: see IMPROVEMENT report / FINAL fair-LS aggregates
- Source raw: `results/raw/cost_aware/IMPROVEMENT_bks_gaps_corrected.csv`

| Method | Mean lit gap % | Median lit gap % | Mean naive gap % (full vs BKS) | Median naive gap % | N instances |
|---|---|---|---|---|---|
| Greedy + Local Search | 7.09 | 4.90 | 44.83 | 39.95 | 20 |
| State-only Pointer | 6.04 | 4.61 | 43.78 | 39.97 | 20 |
| History Transformer | 4.99 | 2.81 | 42.73 | 38.65 | 20 |
| DeepSets Fleet Model | 6.26 | 6.05 | 44.00 | 41.04 | 20 |
| History Transformer + DeepSets | 5.44 | 2.81 | 43.18 | 38.65 | 20 |

## Interpretation

Cite **lit gap** columns only. Naive gaps (~43%) wrongly compare full duration to literature BKS. Each row summarizes 20 benchmark instances.
