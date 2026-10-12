# HAT-EVRP-NL — Final authoritative report

**Status:** Project closed. Architecture and Cycle-3 checkpoints unchanged.  
**Validation:** Independent Python forward simulation (`check_route` / `check_solution`) only. No Java / external DIMACS checker was run.

## Correctness notes (retained)

1. Classical wall-clock RNG fixed via `deterministic_first` (superseded invalid file: `INVALID_wallclock_fair_seed42_c3.md`).
2. Fair LS = best-of-N constructions, then **one** shared local search. Legacy `eqls` (LS on every neural sample) is **not** used in final conclusions.
3. Paired comparisons use the **20 distinct instances** as units (seeds averaged within instance). Tie tolerance \(10^{-9}\). Wins + ties + losses = 100%.

## Architecture

Shared causal visit-history Transformer → dynamic state fusion → DeepSets → hierarchical route/customer decisions → `frvcpy` nonlinear charging → Python feasibility validation. Ablations H1–H4 unchanged. **H2 is the strongest efficient neural variant; H4 remains the full proposed architecture.**

## Protocol A — Equal constructions (no LS)

Raw: `compare_size10_seed{42,43,44}_c3.csv` (60/60 feasible).

| Method | Mean h | Median h | Routes | Charge stops | Charge h | Runtime s | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 22.063 | 22.698 | 2.867 | 4.300 | 1.863 | 0.117 | 315 |
| H0_strong | **20.142** | 18.586 | 2.800 | 3.650 | 1.535 | 0.176 | 528 |
| H1 | 21.249 | 19.333 | 3.000 | 3.900 | 1.761 | 0.460 | 515 |
| H2 | 20.966 | 19.627 | 3.000 | 3.867 | 1.659 | 0.646 | 520 |
| H3 | 22.131 | 21.060 | 3.000 | 4.083 | 1.871 | 0.490 | 497 |
| H4 | 20.975 | 19.185 | 3.100 | 3.717 | 1.641 | 0.671 | 525 |

Instance-paired (`construction_only`): H2 vs H0_strong 20%/0%/80%; H4−H2 40%/25%/35%.

## Protocol B — Fair one-shot local search (primary)

Raw: `compare_size10_fairls_final_seed{42,43,44}_c3.csv` (60/60 feasible).

| Method | Mean h | Median h | Routes | Charge stops | Charge h | Runtime s | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 22.063 | 22.698 | 2.867 | 4.300 | 1.863 | 0.107 | 315 |
| H0_strong | 20.142 | 18.586 | 2.800 | 3.650 | 1.535 | 0.156 | 528 |
| H1 | 19.993 | 18.429 | 2.950 | 3.433 | 1.485 | 0.555 | 750 |
| **H2** | **19.813** | **18.378** | 2.917 | 3.433 | 1.465 | 0.741 | 747 |
| H3 | 20.109 | 18.445 | 2.933 | 3.517 | 1.534 | 0.592 | 729 |
| H4 | 19.886 | 18.586 | 3.033 | 3.350 | 1.414 | 0.753 | 753 |

### Instance-paired comparisons (fair_one_shot_ls, n=20, seed-averaged)

| Pair | Mean Δ (h) | 95% CI | Wins | Ties | Losses |
|---|---:|---|---:|---:|---:|
| H2−H1 | −0.180 | [−0.317, −0.044] | 80% | 10% | 10% |
| H3−H1 | +0.116 | [−0.151, +0.384] | 25% | 5% | 70% |
| **H4−H2** | **+0.074** | **[−0.018, +0.165]** | **20%** | **60%** | **20%** |
| H2−H0_strong | −0.329 | [−0.699, +0.041] | **65%** | **20%** | **15%** |
| H4−H0_strong | −0.255 | [−0.631, +0.120] | **50%** | **20%** | **30%** |

**H4 does not outperform H2** (mean slightly worse; mostly ties). History helps (H2≪H1). DeepSets alone does not (H3). With fair LS, H2 improves on H0_strong on 65% of instances (CI on mean Δ includes values near zero). Neural uses ~1.4× oracle calls and ~4–5× runtime vs H0_strong.

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

H2 remains best among these under matched measured time; H4 does not beat H2.

## Scale (fair LS, seed 42) — verified from raw CSVs

| Size | H0_strong | H2 | H4 | Feasible |
|---|---:|---:|---:|---|
| n=20 | 34.420 | 32.332 | 32.322 | 20/20 |
| n=40 | 60.195 | 57.173 | 57.506 | 20/20 |
| n=80 (3 inst.) | 95.506 | 88.151 | 88.151 | 3/3 |

160/320 not evaluated.

## BKS gaps (Froger et al. 2021 Table 13; n=10 fair LS)

| Method | Mean gap % | Median gap % |
|---|---:|---:|
| H0_strong | 44.8 | 40.0 |
| H2 | 42.7 | 38.7 |
| H4 | 43.2 | 38.7 |

Not competitive with specialized metaheuristics. BKS are not claimed as proven optima.

## Direct CHARGE

4/20 feasible (negative ablation). Hybrid neural routing + `frvcpy`: 20/20. Repair ≠ learned charging.

## Independent Python validation

- `evaluate.py --mode sanity` on `tc0c10s2cf1` executed.
- Representative H2/H4 `c3` decode + one-shot LS: `check_solution` passed; objective equals sum of independently simulated route durations.
- Aggregates in `FINAL_*.csv` recomputed from raw experiment CSVs.
- **This is Python forward-simulation validation, not external checker certification.**

## Scientific conclusion

1. Transformer visit history improves neural routing (H2 vs H1).
2. DeepSets does not consistently help (H4 ≉ better than H2; H3 ≉ better than H1).
3. With fair one-shot LS (and corrected wall-clock), history-aware H2 is the strongest/most efficient neural variant and can beat the tested classical greedy+LS baseline on n=10–40, at higher compute cost and large BKS gaps.
4. Hybrid `frvcpy` charging is required; direct learned charging is unreliable.

**The DeepSets coordination hypothesis is not supported. HAT-EVRP-NL remains a valid, reproducible hybrid research implementation with honest negative findings on DeepSets and learned charging.**

## Invalid / superseded (do not cite as final)

- `*_eqls_*` (unequal per-sample LS)
- `compare_size10_wallclock_fair_seed42_c3.csv` (non-randomized classical trials)
