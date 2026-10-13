# HAT-EVRP-NL — Bounded performance-improvement report

**Date:** 2026-10-13  
**Base commit:** `028e13d` (frozen manuscript)  
**Constraint:** original Transformer+DeepSets architecture retained; Cycle-3 checkpoints not overwritten; no Java; no retraining sweep.

## P1 — Literature objective / BKS correction — RETAIN

### Finding

Montoya et al. (2017) minimize **travel + charging** in the objective, while including service time only in the \(T_{\max}\) constraint (example: route duration 19.2 with service vs cost 18.2 without). Official instances use 0.5 h service per customer. The Froger et al. (2022) Table 13 BKS for `tc2c40s5cf3` is 19.65 h with 40×0.5 = 20 h unavoidable service—impossible if BKS included service.

Our simulator correctly includes service in feasibility / \(T_{\max}\) / full duration. Prior ~43% gaps compared **full duration** to **literature BKS** (service-excluded).

### Dual reporting

- `objective_h`: travel + service + charging (feasibility / optimization in this repo).
- `objective_lit_h`: full − Σ customer service (literature-compatible when all customers served once).

Helpers: `charging.literature_compatible_objective_hours`, `total_customer_service_hours`.

### Corrected n=10 fair-LS gaps vs Froger Table 13 (uncapacitated)

| Method | Naive gap (full vs BKS) | Corrected gap (lit vs BKS) |
|---|---:|---:|
| H0_strong | 44.8% | **7.1%** |
| H2 | 42.7% | **5.0%** |
| H4 | 43.2% | **5.4%** |

Files: `IMPROVEMENT_bks_gaps_corrected.csv`, `IMPROVEMENT_bks_gaps_corrected_detail.csv`.  
Original `FINAL_n10_bks_gaps.csv` preserved (naive).

**Retain.** Rankings among methods on the same instances are unchanged by subtracting a constant service total; only literature comparisons change.

---

## P2 — Oracle-aware customer scoring — RETAIN

### Method

Inference-only: `AdjustedScore_j = logit_j − λ · (D(R⊕j)−D(R))/T_max`, using FRVCP durations already computed for feasibility masks (`train.decode(..., cost_lambda=...)`). `λ=0` = Cycle-3 baseline.

### Synthetic validation (8 instances, seed 42 checkpoints, fair LS)

| Method | Best λ | Synth mean (h) | vs λ=0 |
|---|---:|---:|---:|
| H2 | 2.0 | 13.0635 | −0.006 |
| H4 | 4.0 | 13.0239 | −0.059 |

### Official n=10 fair one-shot LS (seeds 42–44, same LS budget)

| Setting | Mean full h | Lit gap mean | Oracle/inst | Runtime s |
|---|---:|---:|---:|---:|
| H2 Cycle-3 (λ=0) | 19.813 | 5.0% | 747 | 0.74 |
| **H2 λ=2** | **19.706** | **3.9%** | 749 | 0.50 |
| H4 Cycle-3 (λ=0) | 19.886 | 5.4% | 753 | 0.75 |
| **H4 λ=4** | **19.810** | **4.6%** | 759 | 0.65 |

Instance-paired (seed-averaged): H2 λ=2 vs λ=0 → mean Δ −0.107 h (7/8/5 win/tie/loss). H4 λ=4 → −0.076 h (11/5/4).

Files: `IMPROVEMENT_costaware_synth_tune.csv`, `IMPROVEMENT_costaware_H2_n10_lam2p0_fairls.csv`, `IMPROVEMENT_costaware_H4_n10_lam4p0_fairls.csv`.

**Retain** λ=2 (H2) and λ=4 (H4) as optional inference defaults for new runs. Does not replace Cycle-3 tables as historical baseline.

---

## P3 — Stronger local search — RETAIN (with cost note)

### Method

`solver.local_search(..., operators='strong')` adds, under the **same** FRVCP evaluation budget (3 passes / 4000 evals): inter-route customer swap, 2-opt* tail exchange, route elimination with greedy reinsertion. `operators='basic'` = Cycle-3 2-opt+relocate.

Applied identically after best-of-N to H0_strong, H2 (λ=2), H4 (λ=4).

### Official n=10 (seeds 42–44)

| Method | Operators | Mean full h | Lit gap | Oracle/inst | Runtime s |
|---|---|---:|---:|---:|---:|
| H0_strong | basic (Cycle-3) | 20.142 | 7.1% | 528 | 0.16 |
| H0_strong | **strong** | **19.751** | **4.3%** | 779 | 0.13 |
| H2 λ=2 | basic | 19.706 | 3.9% | 749 | 0.50 |
| H2 λ=2 | **strong** | **19.487** | **2.4%** | 1029 | 0.65 |
| H4 λ=4 | basic | 19.810 | 4.6% | 759 | 0.65 |
| H4 λ=4 | **strong** | **19.407** | **1.5%** | 1048 | 0.74 |

### n=20 seed 42 (transfer from n=10 checkpoints)

| Method | Cycle-3 fair LS | Cost-aware + strong LS |
|---|---:|---:|
| H0_strong | 34.420 | **30.652** |
| H2 | 32.332 | **30.409** |
| H4 | 32.322 | **30.279** |

Files: `IMPROVEMENT_strong_ls_n10_fairls.csv`, `IMPROVEMENT_strong_ls_n20_seed42.csv`.

**Retain** strong operators as optional. Neural methods use more oracle calls under the same evaluation *budget* because construction already consumes calls; document this when claiming fair search budgets.

---

## P4 — Teacher / imitation changes — NOT RUN

Inference/search improvements already moved quality; no controlled evidence that imitation labeling is the remaining bottleneck. Deferred.

---

## Recommended configuration for new evaluations

1. Report **both** `objective_h` and `objective_lit_h`; use lit for BKS gaps.
2. Decode with `cost_lambda=2` (H2) or `4` (H4).
3. Fair LS with `operators='strong'` at equal passes/budget.
4. Keep Cycle-3 λ=0 / basic-LS tables as the frozen historical baseline.

## What was rejected / not claimed

- No architecture change; no Java checker; no official-instance training.
- Cost-aware gains are real but modest alone; strongest gains come from **search neighborhoods + corrected BKS accounting**.
- Still not SOTA vs specialized E-VRP-NL metaheuristics; corrected H2 lit gap ~2.4% (strong) is far closer than the erroneous 43% figure, but n=40/80/160 not fully re-benchmarked under the new stack.
