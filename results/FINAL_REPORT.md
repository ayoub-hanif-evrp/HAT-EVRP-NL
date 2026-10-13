# HAT-EVRP-NL — Final authoritative numerical report

**Status:** Research frozen. Architecture and Cycle-3 checkpoints unchanged.  
**Manuscript:** [`../paper_draft.md`](../paper_draft.md)  
**Figures:** [`figures/`](figures/)  
**Validation:** Independent Python forward simulation (`check_route` / `check_solution`) only. No Java / external DIMACS checker was run.

## Correctness notes

1. Classical wall-clock RNG fixed via `deterministic_first` (superseded: `INVALID_wallclock_fair_seed42_c3.md`).
2. Fair LS = best-of-N constructions, then **one** shared local search. Legacy `eqls` is **not** final.
3. Paired comparisons: **20 instances** as units (seeds averaged). Tie tolerance \(10^{-9}\). Wins+ties+losses = 100%.
4. **Protocol A is not a uniform no-LS table:** neural rows are construction-only; **H0_strong still uses local search** and is a stronger reference baseline.

## Architecture

Shared causal visit-history Transformer → dynamic state fusion → DeepSets → hierarchical route/customer decisions → `frvcpy` → Python feasibility validation.  
**H2** = strongest efficient neural variant; **H4** = complete proposed architecture (does **not** consistently beat H2).

## Protocol A — Equal construction counts (neural without LS)

Raw: `compare_size10_seed{42,43,44}_c3.csv` (60/60 feasible).

| Method | Mean h | Median h | Routes | Charge stops | Charge h | Runtime s | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 22.063 | 22.698 | 2.867 | 4.300 | 1.863 | 0.117 | 315 |
| H0_strong† | **20.142** | 18.586 | 2.800 | 3.650 | 1.535 | 0.176 | 528 |
| H1 | 21.249 | 19.333 | 3.000 | 3.900 | 1.761 | 0.460 | 515 |
| H2 | 20.966 | 19.627 | 3.000 | 3.867 | 1.659 | 0.646 | 520 |
| H3 | 22.131 | 21.060 | 3.000 | 4.083 | 1.871 | 0.490 | 497 |
| H4 | 20.975 | 19.185 | 3.100 | 3.717 | 1.641 | 0.671 | 525 |

†Includes local search. Instance-paired (`construction_only`): H2 vs H0_strong 20%/0%/80%; H4−H2 40%/25%/35%.

## Protocol B — Fair one-shot local search (primary)

Raw: `compare_size10_fairls_final_seed{42,43,44}_c3.csv` (60/60 feasible). Aggregate: `FINAL_n10_fairls.csv`.

| Method | Mean h | Median h | Routes | Charge stops | Charge h | Runtime s | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 22.063 | 22.698 | 2.867 | 4.300 | 1.863 | 0.107 | 315 |
| H0_strong | 20.142 | 18.586 | 2.800 | 3.650 | 1.535 | 0.156 | 528 |
| H1 | 19.993 | 18.429 | 2.950 | 3.433 | 1.485 | 0.555 | 750 |
| **H2** | **19.813** | **18.378** | 2.917 | 3.433 | 1.465 | 0.741 | 747 |
| H3 | 20.109 | 18.445 | 2.933 | 3.517 | 1.534 | 0.592 | 729 |
| H4 | 19.886 | 18.586 | 3.033 | 3.350 | 1.414 | 0.753 | 753 |

### Instance-paired (`fair_one_shot_ls`, n=20)

| Pair | Mean Δ (h) | 95% CI | Wins | Ties | Losses |
|---|---:|---|---:|---:|---:|
| H2−H1 | −0.180 | [−0.317, −0.044] | 80% | 10% | 10% |
| H3−H1 | +0.116 | [−0.151, +0.384] | 25% | 5% | 70% |
| **H4−H2** | **+0.074** | **[−0.018, +0.165]** | **20%** | **60%** | **20%** |
| H2−H0_strong | −0.329 | [−0.699, +0.041] | **65%** | **20%** | **15%** |
| H4−H0_strong | −0.255 | [−0.631, +0.120] | **50%** | **20%** | **30%** |

**H4 does not outperform H2.** History helps; DeepSets does not. Neural ≈1.4× oracle and ≈4–5× runtime vs H0_strong.

## Protocol C — Corrected equal wall-clock (seed 42)

Raw: `compare_size10_wallclock_final_seed42_c3.csv`. Unique constructions: H0 ≈47.9, H0_strong ≈40.7.

| Method | Mean h | Runtime s | Oracle |
|---|---:|---:|---:|
| H0 | 20.758 | 0.471 | 2632 |
| H0_strong | 19.895 | 0.412 | 2466 |
| H1 | 20.164 | 0.359 | 775 |
| **H2** | **19.586** | 0.448 | 760 |
| H3 | 20.196 | 0.380 | 771 |
| H4 | 19.840 | 0.458 | 745 |

## Scale (fair LS)

| Size | H0_strong | H2 | H4 | Feasible |
|---|---:|---:|---:|---|
| n=20 | 34.420 | 32.332 | 32.322 | 20/20 |
| n=40 | 60.195 | 57.173 | 57.506 | 20/20 |
| n=80 (3 inst.) | 95.506 | 88.151 | 88.151 | 3/3 |

160/320 not evaluated.

## BKS gaps (Froger et al. 2022 / online 2021 Table 13; uncapacitated; n=10 fair LS)

Provenance: `literature_bks_froger2021.csv` (spot-checked against HAL PDF of Froger et al., *Transportation Science* 2022).

**Convention correction (2026-10-13):** Montoya’s objective is travel+charging; service enters \(T_{\max}\) only. Our `objective_h` is full duration (travel+service+charging). Literature-compatible objective = full − Σ service (n=10: −5.0 h). Naive gaps comparing full duration to BKS are **incorrect**.

| Method | Naive gap (full vs BKS) | Corrected lit gap |
|---|---:|---:|
| H0_strong | 44.8% | **7.1%** |
| H2 | 42.7% | **5.0%** |
| H4 | 43.2% | **5.4%** |

Sources: `FINAL_n10_bks_gaps.csv` (naive, retained), `IMPROVEMENT_bks_gaps_corrected.csv`.  
Further optional inference/search gains: `IMPROVEMENT_REPORT.md` (cost-aware λ + strong LS; not part of frozen Cycle-3 baseline tables above).

### Targeted follow-up (matched λ; see `TARGETED_IMPROVEMENT_REPORT.md`)

Under **identical** `cost_lambda` and strong LS, H4 does **not** reliably beat H2 (layout-clustered CIs include 0). Best verified n=10 config this cycle: **H2 λ=4 + strong LS → 19.335 h** (~0.60% lit BKS gap). Earlier H4-vs-H2 “wins” with λ=2 vs λ=4 were confounded.

## Direct CHARGE

4/20 feasible (`direct_charge_size10_seed42_c3.csv`). Hybrid `frvcpy`: 20/20. Repair ≠ learned charging.

## Independent Python validation

- `evaluate.py --mode sanity` executed.
- Representative H2/H4 `c3` decode + one-shot LS: `check_solution` passed; objective = sum of simulated route durations.
- Final aggregates reproduce raw CSVs; paired W/T/L sum to 100%.
- **Python forward-simulation validation, not external checker certification.**

## Scientific conclusion

1. Transformer history improves neural routing (H2 vs H1).
2. DeepSets does not consistently help (H4 ≉ better than H2).
3. With fair one-shot LS, H2 is the strongest/most efficient neural variant and can beat the tested classical greedy+LS baseline on n=10–40, at higher cost and large BKS gaps.
4. Hybrid `frvcpy` charging is required; direct learned charging is unreliable.

**DeepSets coordination hypothesis: not supported.** Project closed as a reproducible hybrid experimental study with honest negative findings on DeepSets and learned charging.

## Invalid / superseded (do not cite as final)

- `*_eqls_*` (unequal per-sample LS)
- `compare_size10_wallclock_fair_seed42_c3.csv` (non-randomized classical trials)
