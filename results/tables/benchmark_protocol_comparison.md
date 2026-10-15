# Main benchmark comparison (protocol-separated)

All objectives are **full duration hours** \(Z_\mathrm{full}\) unless noted.
Literature gap uses \(Z_\mathrm{lit}=Z_\mathrm{full}-0.5n\) vs Froger et al. 2021 Table 13.
**Do not rank methods across protocol blocks.**

Display names follow `presentation_names.py`. Internal checkpoint keys (H0–H4) appear only in raw CSV filenames.

## Block A — n=10 Cycle-3 fair one-shot LS (matched)

Source: `results/raw/cycle3/FINAL_n10_fairls.csv`  
N = 20 instances × 3 training seeds = 60 evaluations/method; all feasible.

| Method | Mean \(Z_\mathrm{full}\) (h) | Mean lit. gap % | Mean runtime (s) | Mean oracle calls |
|---|---:|---:|---:|---:|
| Greedy Construction | 22.063 | — | 0.11 | 315 |
| Greedy + Local Search | 20.142 | ≈7.09 | 0.16 | 528 |
| State-only Pointer | 19.993 | ≈6.04 | 0.56 | 750 |
| History Transformer | **19.813** | **≈4.99** | 0.74 | 747 |
| DeepSets Fleet Model | 20.109 | ≈6.26 | 0.59 | 729 |
| History Transformer + DeepSets | 19.886 | ≈5.44 | 0.75 | 753 |

## Block B — n=10 insertion / hybrid repair (more expensive)

Source: `results/raw/insertion_repair/INSERTION_official_n10.csv`  
N = 20 instances; single reported evaluation seed family per checkpoint tag.

| Method / mode | Mean \(Z_\mathrm{full}\) (h) | Mean lit. gap % | Mean oracle | Mean runtime (s) |
|---|---:|---:|---:|---:|
| History Transformer + hybrid insert | **19.258** | **≈0.027** | 1140 | 0.77 |
| History Transformer + LS | 19.266 | ≈0.097 | 839 | 0.53 |
| History + DeepSets + hybrid insert | 19.570 | ≈3.45 | 1150 | 0.77 |

This block is **not** an equal-budget match to Block A. The ~0.03% History Transformer hybrid gap must not replace the ~5% fair-LS figure as “the” n10 result without protocol disclosure.

## Block C — n=40 recovered hybrid_insert

Source: `results/raw/regression_recovery/RECOVERY_n40_recovered_pipeline.csv`  
N = 20 instances; recovered pipeline (single evaluation seed context).

| Method | Mean \(Z_\mathrm{full}\) (h) | Mean lit. gap % | Median lit. gap % | Feasible |
|---|---:|---:|---:|---|
| History Transformer | 56.806 | 17.72 | 17.23 | 20/20 |
| History Transformer + DeepSets | 56.370 | 16.79 | 16.78 | 20/20 |

Paired (History+DeepSets − History): mean −0.436 h; 7 wins / 4 ties / 9 losses; layout CI includes 0.

## Machine-readable

CSV companion written by the audit script when available:
`results/tables/benchmark_protocol_comparison.csv`
(Internal method keys retained in the CSV for joinability with raw experiment files.)
