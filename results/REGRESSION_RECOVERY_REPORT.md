# Regression Recovery Report

Critical finding: latest insertion-hybrid n40 results beat their *own*
classical baseline but are **worse than historical fair-LS** on every
instance. This audit recovers the old protocol and re-applies hybrid search.

## 1. Historical vs latest insertion pipeline (same 20 instances)

| Method | Mode | Mean hist h | Mean insertion h | Mean Δ h | Worse count |
|---|---|---:|---:|---:|---:|
| H2 | classical_ruin | 57.1727 | 65.2301 | +8.0573 | 20/20 |
| H2 | hybrid_insert | 57.1727 | 63.8320 | +6.6593 | 20/20 |
| H2 | ls | 57.1727 | 69.0400 | +11.8672 | 20/20 |
| H4 | classical_ruin | 57.5056 | 68.3881 | +10.8824 | 20/20 |
| H4 | hybrid_insert | 57.5056 | 66.8785 | +9.3728 | 20/20 |
| H4 | ls | 57.5056 | 70.9552 | +13.4496 | 20/20 |

## 2. Protocol differences

| Aspect | Historical fair-LS | Insertion experiment |
|---|---|---|
| Construction samples | 3 | 4 (n40) / 3–6 other |
| First attempt | greedy (`sample=False`) | always `sample=True` |
| Start diversity | none | forced first customer |
| cost_lambda | **0** | **4** |
| LS operators | basic | strong |
| Post-process | one-shot LS | LS + ruin / hybrid |

## 3. Construction ablation (identical basic one-shot LS)

| Config | H2 mean LS h | H4 mean LS h |
|---|---:|---:|
| ordinary_l0 | 57.1133 | 57.4126 |
| ordinary_l4 | 58.8275 | 58.1462 |
| forced_l0 | 71.3874 | 72.9465 |
| forced_l4 | 67.4623 | 69.1329 |

Historical targets: H2=57.1727, H4=57.5056.

## 4. Recovered ordinary λ=0 construction + staged search

| Method | Stage | Mean h | Mean Δ vs hist | Beats hist | Mean exec |
|---|---|---:|---:|---:|---:|
| H2 | basic_ls | 57.1133 | -0.0595 | 4/20 | 4656 |
| H2 | classical_ruin | 57.0343 | -0.1385 | 8/20 | 2816 |
| H2 | construct | 59.7172 | +2.5445 | 0/20 | 2424 |
| H2 | hybrid_insert | 56.8055 | -0.3672 | 13/20 | 3571 |
| H2 | neural_insert | 57.6935 | +0.5208 | 3/20 | 3256 |
| H2 | strong_ls | 57.8618 | +0.6890 | 1/20 | 5677 |
| H4 | basic_ls | 57.4126 | -0.0931 | 7/20 | 4657 |
| H4 | classical_ruin | 56.5603 | -0.9453 | 12/20 | 2797 |
| H4 | construct | 59.7893 | +2.2836 | 0/20 | 2400 |
| H4 | hybrid_insert | 56.3700 | -1.1357 | 13/20 | 3550 |
| H4 | neural_insert | 57.7121 | +0.2065 | 9/20 | 3229 |
| H4 | strong_ls | 57.9364 | +0.4307 | 2/20 | 5654 |

## 5. Fair search from recovered strong-LS starts

| Method | Mode | Mean h | Mean exec | Mean search s |
|---|---|---:|---:|---:|
| H2 | classical_ruin | 57.0343 | 392 | 0.153 |
| H2 | hybrid_insert | 56.8055 | 1147 | 1.081 |
| H2 | neural_insert | 57.6935 | 832 | 0.828 |
| H2 | strong_ls | 57.8618 | 3253 | 0.000 |
| H4 | classical_ruin | 56.5603 | 397 | 0.155 |
| H4 | hybrid_insert | 56.3700 | 1149 | 1.086 |
| H4 | neural_insert | 57.7121 | 828 | 0.867 |
| H4 | strong_ls | 57.9364 | 3253 | 0.000 |

## 6. Non-regression check

- Protocol recovery: ordinary starts, `cost_lambda=0`, 3 samples (greedy then sample), basic one-shot LS.
- H2 recovered basic_ls mean **57.113** vs hist **57.173** (16/20 exact instance matches).
- H4 recovered basic_ls mean **57.413** vs hist **57.506** (13/20 exact).
- Small mean differences are within numerical/nondeterminism noise; recovered is slightly *better*.
- Tolerance: pass if recovered mean ≤ historical + 0.15 h.

## 7. Root cause (measured)

| Factor | H2 mean after basic LS | Effect |
|---|---:|---|
| ordinary + λ=0 (historical) | **57.11** | baseline recovered |
| ordinary + λ=4 | 58.83 | mild harm (~+1.7 h) |
| forced + λ=0 | 71.39 | **catastrophic** (~+14 h) |
| forced + λ=4 | 67.46 | still catastrophic |

**Primary cause:** forced-first-customer multi-start on every attempt (n10→n40 transfer).
**Secondary:** `cost_lambda=4` trained/tuned around n10 harms n40 ordinary construction.
Position-aware insertion was never the regression; it was applied on already-broken basins.

## 8. Final retained configurations (n40)

| Rank | Config | Mean full h | vs hist H2 57.17 |
|---|---|---:|---|
| 1 | **H4_c3 + ordinary λ=0 + strong LS + hybrid_insert** | **56.370** | beats hist |
| 2 | H4_c3 + ordinary λ=0 + strong LS + classical_ruin | 56.560 | beats hist |
| 3 | **H2_c3 + ordinary λ=0 + strong LS + hybrid_insert** | **56.806** | beats hist |
| 4 | H2_c3 + ordinary λ=0 + strong LS + classical_ruin | 57.034 | beats hist |
| — | Historical fair-LS H2 / H4 | 57.173 / 57.506 | reference |
| — | INSERTION forced+λ4 hybrid H2 / H4 | 63.832 / 66.878 | **do not use** |

- Hybrid beats classical under the recovered construction (matched starts).
- Hybrid uses more oracle executions (~1147 vs ~395 ruin-only); still improves quality.
- Pure neural_insert alone does **not** beat classical ruin.
- Under matched recovered settings, **H4 hybrid beats H2 hybrid** on n40 mean duration.

## 9. Remaining limitations

- Equal wall-clock / equal-oracle-budget trajectories across seeds not fully swept (single seed 42).
- Strong LS alone can be slightly worse than basic LS from the same construction; hybrid/classical recover from the strong-LS incumbent.
- Do not select `cost_lambda` on official n40; n10-tuned λ=4 does not transfer cleanly.

## Commands

```bash
python results/run_regression_recovery.py --stage all
```
