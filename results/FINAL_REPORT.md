# HAT-EVRP-NL — Final authoritative report

**Project closure.** Architecture preserved. Cycle-3 checkpoints reused. No further training.

## Correctness fixes in this closure

1. **Wall-clock classical RNG bug.** `randomized_greedy(restarts=1)` always took the deterministic `restart==0` branch, so every single-restart trial was identical. Fixed via `deterministic_first` (True only for the first trial of a multi-restart call / first wall-clock trial). Wall-clock now records **~38–48 unique constructions per instance**. File `compare_size10_wallclock_fair_seed42_c3.csv` is **INVALID** (see `INVALID_wallclock_fair_seed42_c3.md`).
2. **Equal LS protocol** (retained): best-of-N without LS, then one shared `local_search`. Legacy `eqls` (LS on every neural sample) remains invalid for equal-effort claims.
3. **External Java checker:** not run (explicitly skipped). Independent Python `check_route`/`check_solution` used for all reported feasible solutions.
4. **BKS:** `literature_bks_froger2021.csv` spot-checked against Froger et al. (2021) Table 13 (uncapacitated E-VRP-NL). Gaps use \(100(z-\mathrm{BKS})/\mathrm{BKS}\). These BKS are **not** claimed as proven optima.

## Architecture (unchanged)

Shared causal visit-history Transformer → state fusion → DeepSets → hierarchical routing → `frvcpy` nonlinear charging → independent validation. Ablations H1–H4 unchanged.

## Experimental protocols

| Label | Protocol |
|---|---|
| **A. Equal construction** | N samples, no LS (`compare_size10_seed{42,43,44}_c3.csv`) |
| **B. Fair one-shot LS** | N constructions, one shared LS (`*_fairls_final_*`) |
| **C. Fair wall-clock** | Measured neural time; classical uses same wall time with fixed RNG (`*_wallclock_final_*`) |

Primary paired units: **20 instances** (seeds averaged within instance). Seeds 42/43/44 for n=10. Seed 42 for n=20/40/80 sample.

## Final n=10 results

### A. Equal constructions (60 rows / 20 instances × 3 seeds; all 60/60 feasible)

| Method | Mean h | Oracle/inst |
|---|---|---|
| H0 | 22.063 | 315 |
| H0_strong | **20.142** | 528 |
| H1 | 21.249 | 515 |
| H2 | 20.966 | 520 |
| H3 | 22.131 | 497 |
| H4 | 20.975 | 525 |

History helps (H2≪H1). DeepSets without history hurts/neutral (H3). H4 ≈ H2; both lose to H0_strong.

### B. Fair one-shot LS (corrected equal refinement)

| Method | Mean h | Oracle/inst | Runtime s/inst |
|---|---|---|---|
| H0 | 22.063 | 315 | 0.107 |
| H0_strong | 20.142 | 528 | 0.156 |
| H1 | 19.993 | 750 | 0.555 |
| H2 | **19.813** | 747 | 0.741 |
| H3 | 20.109 | 729 | 0.592 |
| H4 | 19.886 | 753 | 0.753 |

Paired over **20 instances** (seed-averaged):

| Pair | Mean Δ (h) | Left better | Tie | Right better |
|---|---|---|---|---|
| H2−H1 | −0.180 | 80% | 10% | 10% |
| H3−H1 | +0.116 | 25% | 5% | 70% |
| **H4−H2** | **+0.074** | **30%** | **60%** | **20%** |
| H2−H0_strong | −0.329 | 65% | 20% | 20% |
| H4−H0_strong | −0.255 | 55% | 20% | 30% |

**H4 does not improve over H2.** History is useful. With fair one-shot LS, H2/H4 improve on H0_strong on a majority of instances, at ~1.4× oracle calls and ~4–5× runtime.

### C. Fair wall-clock (seed 42, fixed classical RNG)

Mean unique constructions: H0 47.85; H0_strong 40.65. Mean runtimes ≈0.36–0.47 s.

| Method | Mean h |
|---|---|
| H0 | 20.758 |
| H0_strong | 19.895 |
| H2 | **19.586** |
| H4 | 19.840 |

Under matched measured time, H2 still edges H0_strong; H4 is essentially tied with H0_strong and does not beat H2.

## Scale (fair LS, seed 42)

| Size | H0_strong | H2 | H4 | Feasible |
|---|---|---|---|---|
| n=20 | 34.420 | 32.332 | 32.322 | 20/20 |
| n=40 | 60.195 | 57.173 | 57.506 | 20/20 |
| n=80 (3 inst.) | 95.506 | 88.151 | 88.151 | 3/3 |

160/320 not evaluated (CPU budget).

## BKS gaps (Froger 2021 Table 13; n=10 fair LS, seed-averaged)

| Method | Mean gap % | Median gap % |
|---|---|---|
| H0_strong | 44.8 | 40.0 |
| H2 | 42.7 | 38.7 |
| H4 | 43.2 | 38.7 |

Not competitive with specialized ILS/LNS metaheuristics.

## Direct CHARGE

4/20 feasible (negative ablation). Hybrid + `frvcpy`: 20/20. Repair is not learned charging.

## Scientific verdict

1. **Transformer history (H2 vs H1):** clear benefit.
2. **DeepSets alone (H3 vs H1):** no benefit.
3. **DeepSets on top of history (H4 vs H2):** **no meaningful improvement**; often tied, mean slightly worse.
4. **Neural vs strong classical:** construction-only loses to H0_strong; fair one-shot LS and fair wall-clock show H2 can beat H0_strong on this CPU setup, but with more oracle/runtime cost and large BKS gaps.
5. **Learned nonlinear charging:** not reliable; keep hybrid `frvcpy`.

**HAT-EVRP-NL is a valid research implementation of history-aware hybrid E-VRP-NL routing. The DeepSets coordination hypothesis is not supported by the final fair evidence.**

## Commands executed (closure)

```text
# RNG diversity check (15/15 unique constructions)
python -c "..."  # deterministic_first=False

# Final wall-clock + fair LS n=10
python evaluate.py --mode compare --size 10 --limit 20 --samples 10 --seed {42,43,44} \
  --version c3 --local-passes 3 --local-budget 4000 --fair-ls --tag fairls_final
python evaluate.py --mode compare --size 10 --limit 20 --samples 10 --seed 42 \
  --version c3 --local-passes 3 --local-budget 4000 --fair-ls --wallclock --tag wallclock_final

# Scale
python evaluate.py --mode compare --size 20 --limit 20 --samples 5 --seed 42 \
  --checkpoint-size 10 --local-passes 2 --local-budget 3000 --fair-ls --tag fairls_final
python evaluate.py --mode compare --size 40 --limit 20 --samples 3 --seed 42 \
  --checkpoint-size 10 --local-passes 2 --local-budget 3000 --fair-ls --tag fairls_final
python evaluate.py --mode compare --size 80 --limit 3 --samples 2 --seed 42 \
  --checkpoint-size 10 --local-passes 1 --local-budget 1500 --fair-ls --tag fairls_final
```

## Artifacts

- Tables: `FINAL_n10_*.csv`, `FINAL_paired_by_instance.csv`, `FINAL_*_fairls.csv`
- Figures: `FINAL_n10_comparisons.png`, `FINAL_deepsets_paired.png`
- Paper: `../paper_draft.md`
- Checkpoints: `../checkpoints/*_c3.pt` (unchanged)
