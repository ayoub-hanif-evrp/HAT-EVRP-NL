# HAT-EVRP-NL — Final research summary (corrected fairness)

All reported solutions were checked by an independent forward simulator using
Montoya piecewise charging curves, service times, full initial battery, SOC
bounds, exact customer coverage, depot return, and the 10-hour route limit.

Literature BKS values are taken from Froger et al. (2021), Table 13
(`results/literature_bks_froger2021.csv`). The DIMACS Java checker
(https://github.com/ncabrera10/SolutionChecker_E-VRP) was cloned under
`tools/` but **could not be executed** here (no Java runtime). Cross-check:
the official sample MIP XML for `tc2c10s2cf0` is feasible under our simulator
(duration 26.773876 h). Exported DIMACS XML: `results/dimacs_xml/`.

## Architecture (unchanged)

Shared causal EV visit-history Transformer → dynamic state fusion → DeepSets
`g=ρ(∑φ(v_i))` → hierarchical route/customer decoder → `frvcpy` nonlinear
charging → independent validation.

## Critical correction: equal local-search protocol

Earlier Cycle-3 files tagged `eqls` applied local search **to every neural
sample**, while H0_strong applied LS **once after best-of-N**. That inflated
neural oracle effort (~2704 vs ~528 calls) and must **not** be cited as an
equal-effort comparison. Those CSVs are retained only as legacy artifacts.

**Correct fair LS protocol (`--fair-ls`, tag `fairls`):**

1. Generate N constructions for every method **without** local search.
2. Keep the best feasible construction.
3. Apply the **identical** `local_search` once with the same passes/budget.

## Main n=10 results (20 instances × 3 seeds = 60)

### A. Equal construction samples (no neural LS)

| Method | Feasible | Mean h | Oracle/inst |
|---|---|---|---|
| H0 | 60/60 | 22.063 | 315 |
| H0_strong | 60/60 | 20.142 | 528 |
| H1 | 60/60 | 21.249 | 515 |
| H2 | 60/60 | 20.966 | 520 |
| H3 | 60/60 | 22.131 | 497 |
| H4 | 60/60 | 20.975 | 525 |

H4−H2 mean delta +0.009 h (H4 better on 33%). H4 loses to H0_strong
(+0.833 h; win rate 12%). History helps; DeepSets does not.

### B. Fair one-shot local search (corrected)

| Method | Feasible | Mean h | Oracle/inst | Runtime s/inst |
|---|---|---|---|---|
| H0 | 60/60 | 22.063 | 315 | 0.066 |
| H0_strong | 60/60 | 20.142 | 528 | 0.096 |
| H1 | 60/60 | 19.993 | 750 | 0.364 |
| H2 | 60/60 | **19.813** | 747 | 0.476 |
| H3 | 60/60 | 20.109 | 729 | 0.389 |
| H4 | 60/60 | 19.886 | 753 | 0.484 |

Paired (fair LS): H2−H0_strong −0.329 h (H2 better on **60%**);
H4−H0_strong −0.255 h (H4 better on **60%**);
H4−H2 +0.074 h (H4 better on only **18%**).

Oracle effort is now comparable (~1.4× neural vs H0_strong), not ~5× as in
the legacy unequal protocol.

### C. Fair measured wall-clock (seed 42)

Classical constructions continue until measured time matches the neural
instance wall time; then one shared LS (same budget). Mean runtimes ≈0.36–0.50 s.

| Method | Mean h | Runtime s | Samples used |
|---|---|---|---|
| H0 | 23.327 | 0.505 | 51.5 |
| H0_strong | 20.889 | 0.442 | 43.5 |
| H2 | **19.586** | 0.482 | 10 |
| H4 | 19.840 | 0.486 | 10 |

Under matched wall time, history-aware neural+one LS still beats H0_strong on
this seed; DeepSets (H4) does not beat H2.

## Scalability (seed 42, fair LS)

| Size | H0_strong | H2 | H4 | Feasible |
|---|---|---|---|---|
| n=20 | 34.420 | 32.332 | 32.322 | 20/20 |
| n=40 | 60.195 | 57.173 | 57.506 | 20/20 |

## Literature BKS gaps (Froger 2021, seed-42 fair LS)

Mean % gap to published BKS on n=10: H0_strong 47.1%; H2 41.0%; H4 42.7%.
All methods remain far from literature metaheuristics; gaps are reported, not
hidden.

## DeepSets

Permutation-invariant with nonzero gradients, but **no consistent objective
gain over H2**. Fair LS and wall-clock both favor H2 slightly over H4.

## Learned charging

Direct CHARGE remains 4/20 feasible (negative ablation). Hybrid + `frvcpy`
is the validated method.

## Artifacts

- Fair CSVs: `compare_size*_fairls_*_c3.csv`, `compare_size10_wallclock_fair_seed42_c3.csv`
- Legacy unequal LS (do not cite as fair): `compare_size*_eqls_*_c3.csv`
- BKS table: `literature_bks_froger2021.csv`
- Paper draft: `paper_draft.md` (updated)
- Checkpoints unchanged: `checkpoints/*_c3.pt`
