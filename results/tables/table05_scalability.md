# Scalability across problem sizes

- Objective convention: Full duration
- Units: hours
- Benchmark size: 10/20/40/80
- Instance count: 20
- Methods / checkpoints: History Transformer / History + DeepSets
- Search protocol: protocol labeled per row — do not pool blindly
- Seeds: 42 (+ multi-seed n10 where noted)
- Source raw: `FINAL_n*_fairls + RECOVERY pipeline`

| n | Method | Mean full h | Feasible | Protocol |
|---|---|---|---|---|
| 10 | History Transformer | 19.8127 | 60/60 | fair-LS from10 |
| 10 | History Transformer + DeepSets | 19.8863 | 60/60 | fair-LS from10 |
| 20 | History Transformer | 32.3320 | 20/20 | fair-LS from10 |
| 20 | History Transformer + DeepSets | 32.3223 | 20/20 | fair-LS from10 |
| 40 | History Transformer | 57.1727 | 20/20 | fair-LS from10 |
| 40 | History Transformer + DeepSets | 57.5056 | 20/20 | fair-LS from10 |
| 80 | History Transformer | 88.1507 | 3/3 | fair-LS from10 |
| 80 | History Transformer + DeepSets | 88.1507 | 3/3 | fair-LS from10 |
| 40 | History Transformer | 56.8055 | 20/20 | recovered hybrid |
| 40 | History Transformer + DeepSets | 56.3700 | 20/20 | recovered hybrid |

## Interpretation

Training size is n=10 for all rows. Test size varies. Recovered hybrid is a different search protocol than fair-LS. n=80 currently uses only 3 instances — exploratory.
