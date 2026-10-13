# Scalability across problem sizes

- Objective convention: Full duration
- Units: hours
- Benchmark size: 10/20/40/80
- Instance count: 20
- Methods / checkpoints: H2/H4
- Search protocol: protocol labeled per row — do not pool blindly
- Seeds: 42 (+ multi-seed n10 where noted)
- Source raw: `FINAL_n*_fairls + RECOVERY pipeline`

| n | Method | Mean full h | Feasible | Protocol |
|---|---|---|---|---|
| 10 | H2 | 19.8127 | 60/60 | fair-LS from10 |
| 10 | H4 | 19.8863 | 60/60 | fair-LS from10 |
| 20 | H2 | 32.3320 | 20/20 | fair-LS from10 |
| 20 | H4 | 32.3223 | 20/20 | fair-LS from10 |
| 40 | H2 | 57.1727 | 20/20 | fair-LS from10 |
| 40 | H4 | 57.5056 | 20/20 | fair-LS from10 |
| 80 | H2 | 88.1507 | 3/3 | fair-LS from10 |
| 80 | H4 | 88.1507 | 3/3 | fair-LS from10 |
| 40 | H2 | 56.8055 | 20/20 | recovered hybrid |
| 40 | H4 | 56.3700 | 20/20 | recovered hybrid |

## Interpretation

Training size is n=10 for all rows. Test size varies. Recovered hybrid is a different search protocol than fair-LS.
