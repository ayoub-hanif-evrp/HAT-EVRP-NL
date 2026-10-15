# Main model comparison (n=10 fair-LS Cycle-3)

- Objective convention: Full duration (travel+service+charging)
- Units: hours
- Benchmark size: 10
- Instance count: 20 instances × up to 3 seeds (see feasible column)
- Methods / checkpoints: Greedy / State-only / History / DeepSets / History+DeepSets (Cycle-3 n10)
- Search protocol: best-of-10 constructions + one-shot basic LS
- Seeds: 42–44 aggregated where present
- Source raw: `results/raw/cycle3/FINAL_n10_fairls.csv`

| Method | Mean full h | Feasible | Protocol |
|---|---|---|---|
| Greedy Construction | 22.0627 | 60/60 | Cycle-3 fair-LS |
| Greedy + Local Search | 20.1416 | 60/60 | Cycle-3 fair-LS |
| State-only Pointer | 19.9931 | 60/60 | Cycle-3 fair-LS |
| History Transformer | 19.8127 | 60/60 | Cycle-3 fair-LS |
| DeepSets Fleet Model | 20.1094 | 60/60 | Cycle-3 fair-LS |
| History Transformer + DeepSets | 19.8863 | 60/60 | Cycle-3 fair-LS |

## Interpretation

History Transformer is the strongest efficient neural ablation under matched fair-LS; History Transformer + DeepSets (HAT-EVRP-NL) does not show a consistent multi-seed DeepSets win. Do not mix with hybrid-search rows. Internal keys H0–H4 retained in raw CSVs.
