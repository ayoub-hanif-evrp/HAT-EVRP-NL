# Neural construction vs post-search contribution

- Objective convention: Full duration
- Units: hours
- Benchmark / data: n=10 fair/equal; n=40 recovered
- Instance / seed count: 20 (protocols labeled per row)
- Methods / checkpoints: Greedy+LS through History+DeepSets; History vs History+DeepSets on n40
- Protocol: Do not pool rows across protocols
- Seeds: Cycle-3 multi-seed n10; n40 seed 42
- Source raw: `FINAL_n10_equal_construction + FINAL_n10_fairls + RECOVERY pipeline`

| Method | Construction-oriented mean h | After search mean h | Δ (search − construction) | Feasible (constr.) | Feasible (search) | Protocol note |
|---|---|---|---|---|---|---|
| Greedy + Local Search | 20.1416 | 20.1416 | +0.0000 | 60/60 | 60/60 | equal construction vs fair-LS; n=10 Cycle-3 |
| State-only Pointer | 21.2491 | 19.9931 | -1.2560 | 60/60 | 60/60 | equal construction vs fair-LS; n=10 Cycle-3 |
| History Transformer | 20.9658 | 19.8127 | -1.1531 | 60/60 | 60/60 | equal construction vs fair-LS; n=10 Cycle-3 |
| DeepSets Fleet Model | 22.1306 | 20.1094 | -2.0212 | 60/60 | 60/60 | equal construction vs fair-LS; n=10 Cycle-3 |
| History Transformer + DeepSets | 20.9745 | 19.8863 | -1.0883 | 60/60 | 60/60 | equal construction vs fair-LS; n=10 Cycle-3 |
| History Transformer | 59.7172 | 56.8055 | -2.9118 | 20/20 | 20/20 | n=40 recovered: construct vs hybrid_insert (search gain) |
| History Transformer + DeepSets | 59.7893 | 56.3700 | -3.4193 | 20/20 | 20/20 | n=40 recovered: construct vs hybrid_insert (search gain) |

## Interpretation

Negative Δ means search improved the objective. Architecture comparisons should use matched construction or matched search rows — never attribute hybrid-search gains to DeepSets.
