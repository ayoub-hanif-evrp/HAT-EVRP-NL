# Literature-compatible BKS gaps (corrected service convention)

- Objective convention: Lit gap uses (full − Σ service) vs Froger Table 13 travel+charging BKS
- Units: % gap
- Benchmark size: n=10
- Instance count: 20 official instances per method (5 method-summary rows — not 5 instances)
- Methods / checkpoints: H0_strong, H1, H2, H3, H4
- Search protocol: fair-LS Cycle-3; service-time correction applied
- Seeds: see IMPROVEMENT report / FINAL fair-LS aggregates
- Source raw: `results/raw/cost_aware/IMPROVEMENT_bks_gaps_corrected.csv`

| Display name | Key | Mean lit gap % | Median lit gap % | Mean naive gap % (full vs BKS) | Median naive gap % | N instances |
|---|---|---|---|---|---|---|
| Randomized Greedy + Local Search (H0_strong) | H0_strong | 7.09 | 4.90 | 44.83 | 39.95 | 20 |
| State-Only Hierarchical Pointer (H1) | H1 | 6.04 | 4.61 | 43.78 | 39.97 | 20 |
| History Transformer Routing (H2) | H2 | 4.99 | 2.81 | 42.73 | 38.65 | 20 |
| DeepSets Fleet Routing (H3) | H3 | 6.26 | 6.05 | 44.00 | 41.04 | 20 |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | H4 | 5.44 | 2.81 | 43.18 | 38.65 | 20 |

## Interpretation

Cite **lit gap** columns only. Naive gaps (~43%) wrongly compare full duration to literature BKS. Each row summarizes 20 benchmark instances.
