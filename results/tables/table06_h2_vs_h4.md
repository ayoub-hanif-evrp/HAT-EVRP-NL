# H2 vs H4 under recovered hybrid_insert (n=40)

- Objective convention: Full duration
- Units: hours
- Benchmark size: 40
- Instance count: 20
- Methods / checkpoints: H2 vs H4 Cycle-3, identical protocol
- Search protocol: ordinary λ=0 + strong LS + hybrid_insert
- Seeds: 42 (single seed)
- Source raw: `results/raw/regression_recovery/RECOVERY_n40_recovered_pipeline.csv`

| Instance | History Transformer (H2) h | HAT-EVRP-NL (H4) h | H4−H2 |
|---|---|---|---|
| tc0c40s5cf0 | 57.3243 | 57.9150 | +0.5907 |
| tc0c40s5cf4 | 57.0618 | 53.6167 | -3.4451 |
| tc0c40s5ct0 | 54.7035 | 54.5298 | -0.1737 |
| tc0c40s5ct4 | 59.5772 | 54.2337 | -5.3435 |
| tc0c40s8cf0 | 54.3198 | 54.3198 | +0.0000 |
| tc0c40s8cf4 | 57.0965 | 53.9168 | -3.1797 |
| tc0c40s8ct0 | 51.0971 | 51.0971 | +0.0000 |
| tc0c40s8ct4 | 57.7360 | 53.9235 | -3.8125 |
| tc1c40s5cf1 | 93.7188 | 93.7188 | +0.0000 |
| tc1c40s5ct1 | 78.2393 | 78.4066 | +0.1674 |
| tc1c40s8cf1 | 71.7056 | 67.5433 | -4.1622 |
| tc1c40s8ct1 | 69.4223 | 70.6301 | +1.2078 |
| tc2c40s5cf2 | 53.6194 | 52.1865 | -1.4330 |
| tc2c40s5cf3 | 40.1442 | 42.2074 | +2.0631 |
| tc2c40s5ct2 | 47.8793 | 49.3736 | +1.4943 |
| tc2c40s5ct3 | 46.2358 | 46.2358 | +0.0000 |
| tc2c40s8cf2 | 51.9641 | 53.2089 | +1.2448 |
| tc2c40s8cf3 | 40.6194 | 42.2074 | +1.5880 |
| tc2c40s8ct2 | 47.4595 | 50.1414 | +2.6819 |
| tc2c40s8ct3 | 46.1860 | 47.9867 | +1.8007 |

## Interpretation

Wins(H4 better)/ties/losses = 7/4/9. Mean H4−H2 = -0.4355 h. Single-seed recovered hybrid; not a general DeepSets claim.
