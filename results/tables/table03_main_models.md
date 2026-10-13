# Main model comparison (n=10 fair-LS Cycle-3)

- Objective convention: Full duration (travel+service+charging)
- Units: hours
- Benchmark size: 10
- Instance count: 20
- Methods / checkpoints: H0–H4 Cycle-3 n10 checkpoints
- Search protocol: best-of-10 constructions + one-shot basic LS
- Seeds: 42–44 aggregated where present
- Source raw: `results/raw/cycle3/FINAL_n10_fairls.csv`

| Method | Mean full h | Feasible | Protocol |
|---|---|---|---|
| H0 | 22.0627 | 60/60 | Cycle-3 fair-LS |
| H0_strong | 20.1416 | 60/60 | Cycle-3 fair-LS |
| H1 | 19.9931 | 60/60 | Cycle-3 fair-LS |
| H2 | 19.8127 | 60/60 | Cycle-3 fair-LS |
| H3 | 20.1094 | 60/60 | Cycle-3 fair-LS |
| H4 | 19.8863 | 60/60 | Cycle-3 fair-LS |

## Interpretation

H2 is the strongest efficient neural ablation; H4 is the full DeepSets architecture. Do not mix with hybrid-search rows.
