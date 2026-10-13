# Literature-compatible BKS gaps (corrected service convention)

- Objective convention: Lit duration = full − Σ service vs Froger Table 13
- Units: % gap / hours
- Benchmark size: primarily n=10
- Instance count: 5
- Methods / checkpoints: as in source CSV
- Search protocol: documented in IMPROVEMENT report
- Seeds: see source
- Source raw: `results/raw/cost_aware/IMPROVEMENT_bks_gaps_corrected.csv`

| method | mean_gap_full_pct | median_gap_full_pct | mean_gap_lit_pct | median_gap_lit_pct | n |
|---|---|---|---|---|---|
| H0_strong | 44.825781474275736 | 39.95195944068533 | 7.08637262279732 | 4.899427014525678 | 20 |
| H1 | 43.78136823394151 | 39.96801469991044 | 6.041959382463088 | 4.6082826631518286 | 20 |
| H2 | 42.73200117508547 | 38.65149562505397 | 4.99259232360705 | 2.8135631149049076 | 20 |
| H3 | 44.00105679735583 | 41.038372326962815 | 6.261647945877411 | 6.050052888483865 | 20 |
| H4 | 43.18415174339934 | 38.65149562505397 | 5.444742891920921 | 2.8135631149049076 | 20 |

## Interpretation

Do not mix full-duration objectives with travel+charging BKS. The obsolete ~43% figure used full duration vs lit BKS.
