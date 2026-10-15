# Main benchmark comparison (protocol-separated)

All objectives are **full duration hours** \(Z_\mathrm{full}\) unless noted.
Literature gap uses \(Z_\mathrm{lit}=Z_\mathrm{full}-0.5n\) vs Froger et al. 2021 Table 13.
**Do not rank methods across protocol blocks.**

## Block A — n=10 Cycle-3 fair one-shot LS (matched)

Source: `results/raw/cycle3/FINAL_n10_fairls.csv`  
N = 20 instances × 3 training seeds = 60 evaluations/method; all feasible.

| Method | Mean \(Z_\mathrm{full}\) (h) | Mean lit. gap % | Mean runtime (s) | Mean oracle calls |
|---|---:|---:|---:|---:|
| Randomized Greedy (H0) | 22.063 | — | 0.11 | 315 |
| Greedy + LS (H0_strong) | 20.142 | ≈7.09 | 0.16 | 528 |
| State-only (H1) | 19.993 | ≈6.04 | 0.56 | 750 |
| History Transformer (H2) | **19.813** | **≈4.99** | 0.74 | 747 |
| DeepSets (H3) | 20.109 | ≈6.26 | 0.59 | 729 |
| HAT-EVRP-NL (H4) | 19.886 | ≈5.44 | 0.75 | 753 |

## Block B — n=10 insertion / hybrid repair (more expensive)

Source: `results/raw/insertion_repair/INSERTION_official_n10.csv`  
N = 20 instances; single reported evaluation seed family per checkpoint tag.

| Method / mode | Mean \(Z_\mathrm{full}\) (h) | Mean lit. gap % | Mean oracle | Mean runtime (s) |
|---|---:|---:|---:|---:|
| H2_c3 + hybrid_insert | **19.258** | **≈0.027** | 1140 | 0.77 |
| H2_c3 + ls | 19.266 | ≈0.097 | 839 | 0.53 |
| H4_c3 + hybrid_insert | 19.570 | ≈3.45 | 1150 | 0.77 |

This block is **not** an equal-budget match to Block A. The ~0.03% H2 gap must not replace the ~5% fair-LS figure as “the” n10 result without protocol disclosure.

## Block C — n=40 recovered hybrid_insert

Source: `results/raw/regression_recovery/RECOVERY_n40_recovered_pipeline.csv`  
N = 20 instances; recovered pipeline (single evaluation seed context).

| Method | Mean \(Z_\mathrm{full}\) (h) | Mean lit. gap % | Median lit. gap % | Feasible |
|---|---:|---:|---:|---|
| History (H2) | 56.806 | 17.72 | 17.23 | 20/20 |
| HAT-EVRP-NL (H4) | 56.370 | 16.79 | 16.78 | 20/20 |

Paired H4−H2: mean −0.436 h; 7 wins / 4 ties / 9 losses; layout CI includes 0.

## Machine-readable

CSV companion written by the audit script when available:
`results/tables/benchmark_protocol_comparison.csv`
