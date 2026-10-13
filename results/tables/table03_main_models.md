# Main model comparison (n=10 fair-LS Cycle-3)

- Objective convention: Full duration (travel+service+charging)
- Units: hours
- Benchmark size: 10
- Instance count: 20 instances × up to 3 seeds (see feasible column)
- Methods / checkpoints: H0–H4 Cycle-3 n10 checkpoints
- Search protocol: best-of-10 constructions + one-shot basic LS
- Seeds: 42–44 aggregated where present
- Source raw: `results/raw/cycle3/FINAL_n10_fairls.csv`

| Display name | Key | Mean full h | Feasible | Protocol |
|---|---|---|---|---|
| Randomized Greedy Routing (H0) | H0 | 22.0627 | 60/60 | Cycle-3 fair-LS |
| Randomized Greedy + Local Search (H0_strong) | H0_strong | 20.1416 | 60/60 | Cycle-3 fair-LS |
| State-Only Hierarchical Pointer (H1) | H1 | 19.9931 | 60/60 | Cycle-3 fair-LS |
| History Transformer Routing (H2) | H2 | 19.8127 | 60/60 | Cycle-3 fair-LS |
| DeepSets Fleet Routing (H3) | H3 | 20.1094 | 60/60 | Cycle-3 fair-LS |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | H4 | 19.8863 | 60/60 | Cycle-3 fair-LS |

## Interpretation

History Transformer Routing (H2) is the strongest efficient neural ablation under matched fair-LS; HAT-EVRP-NL (H4) is the full architecture without a consistent multi-seed DeepSets win. Do not mix with hybrid-search rows.
