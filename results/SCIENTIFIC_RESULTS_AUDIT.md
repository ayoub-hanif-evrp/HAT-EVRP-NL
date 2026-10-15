# Scientific Results Audit

Independently recomputed from authoritative per-instance CSVs (commit baseline `2789a3e`; this document regenerates from raw evidence).

## Objective conventions

| Symbol | Definition |
|---|---|
| \(Z_\mathrm{full}\) | Full route duration including fixed customer service (hours). |
| \(Z_\mathrm{lit}\) | \(Z_\mathrm{full} - 0.5\times n\) (literature-compatible; Froger/Montoya). |
| BKS gap | \(100\times(Z_\mathrm{lit}-\mathrm{BKS})/\mathrm{BKS}\). |
| Reward | \(R=-Z_\mathrm{full}\) if feasible; \(R=-100\) if infeasible. |

---

## A. Nonlinear charging — fixed-route (n=40)

**Source:** `results/raw/nonlinear_charging/fixed_route_charging_comparison.csv`  
**Protocol:** matched customer sequences; `method` starting with `plan_`; 20 official instances.  
**Confirmed by:** `charging_study_summary.csv` (plan rows).

| Model | True-physics feasible | Planner rejected | Planner-OK / physics-fail |
|---|---:|---:|---:|
| TRUE-NL | **20/20** | 0 | 0 |
| LINEAR-SECANT | **11/20** | **8** | **1** |
| LINEAR-EARLY-RATE | **11/20** | **0** | **9** |

**Verdict: STRONG.** Linear surrogates are not interchangeable with Montoya piecewise-linear physics. Secant tends to reject plans; early-rate tends to accept optimistic plans that fail under true physics.

---

## B. Nonlinear charging — end-to-end (n=40)

**Source:** `results/raw/nonlinear_charging/end_to_end_charging_sensitivity.csv`  
**Decision changes:** `end_to_end_decision_changes.csv`

| Model | True-physics feasible |
|---|---:|
| TRUE-NL | **20/20** |
| LINEAR-SECANT | **11/20** |
| LINEAR-EARLY-RATE | **9/20** |

Route sequences vs TRUE-NL change on 30% (SECANT) / 25% (EARLY) of n=40 instances.

**Verdict: STRONG** (routing decisions + charging law interact; distinct from fixed-route).

---

## C. n=10 matched Cycle-3 fair-LS

**Source:** `results/raw/cycle3/FINAL_n10_fairls.csv`  
(recomputed from `compare_size10_fairls_final_seed{42,43,44}_c3.csv`: 20 instances × 3 seeds = 60 evals/method)

| Method | Mean \(Z_\mathrm{full}\) (h) |
|---|---:|
| Randomized Greedy + LS (H0_strong) | **20.142** |
| State-only pointer (H1) | **19.993** |
| History Transformer (H2) | **19.813** |
| DeepSets-only (H3) | **20.109** |
| History + DeepSets / HAT-EVRP-NL (H4) | **19.886** |

**Verdict: MODEST positive for history** vs state-only and greedy+LS. **No consistent DeepSets win** over history alone on this matched protocol.

---

## D. n=40 recovered hybrid

**Source:** `results/raw/regression_recovery/RECOVERY_n40_recovered_pipeline.csv` stage=`hybrid_insert`  
**BKS:** `results/literature_bks_froger2021.csv`

| Method | Mean \(Z_\mathrm{full}\) | Mean lit. BKS gap |
|---|---:|---:|
| H2 | **56.8055 h** | **≈ 17.72%** |
| H4 | **56.3700 h** | **≈ 16.79%** |

Paired H4−H2 (20 layouts): mean **−0.436 h**; wins/ties/losses **7 / 4 / 9**; layout-SE 95% interval **includes zero** \([-1.46, +0.59]\).

**Verdict: INCONCLUSIVE for DeepSets.** H4 has a lower average but not a statistically dependable paired advantage. Do **not** claim a reliable DeepSets improvement.

Objective decomposition (`RECOVERY_n40_objective_decomposition.csv`): H4’s mean edge is mainly **travel**, not charging (same `frvcpy`).

---

## E. Controlled BC vs RL (official n10 construction-only)

**Source:** `results/raw/reward_learning/bc_vs_rl_official_n10_summary.csv`  
**Protocol:** official Montoya n=10; construction-only; 20 instances/seed; seeds 42–44.

| Model | Seeds where RL < BC |
|---|---|
| H2 | **1/3** |
| H4 | **3/3** |

**Verdict: MODEST / protocol-limited.** Not evidence that RL improves final hybrid routing. Distinct from historical Cycle-3 synthetic-validation RL, where H4 endpoints were often worse than best BC.

---

## F. Historical Cycle-3 RL

Cycle-3 training logs select checkpoints by synthetic validation. Controlled RQ4 and Cycle-3 synth-val **must not be pooled**. Paper must keep them separate.

---

## Competitiveness (multi-protocol — do not mix)

| Protocol | Size | Method | Mean lit. gap | Compute note |
|---|---:|---|---:|---|
| Cycle-3 fair-LS | 10 | H2 | **≈ 4.99%** | Matched fair one-shot LS |
| Cycle-3 fair-LS | 10 | H4 | **≈ 5.44%** | Same |
| Insertion hybrid_insert | 10 | H2_c3 | **≈ 0.027%** | **More expensive** hybrid repair |
| Recovered hybrid_insert | 40 | H4 | **≈ 16.79%** | Single recovered pipeline seed |

**Verdict: NOT state-of-the-art as an unqualified claim.** Near-BKS n10 numbers under hybrid insertion are real but **not** the same experiment as ~5% fair-LS. At n=40, gaps remain ~17%.

Equal wall-clock / equal-`frvcpy` budgets that would support an “efficiency” claim for hybrid search were **not** found as a clean matched study. Quality-vs-compute plots must disclose unequal realized budgets.

---

## Claim strength summary

| Topic | Strength |
|---|---|
| Nonlinear vs linear physical feasibility | **Strong** |
| History Transformer vs state-only (n10 fair-LS) | **Modest** |
| DeepSets advantage | **Inconclusive** |
| Controlled RL over BC (construction-only) | **Modest, H4 only; not hybrid proof** |
| Near-zero n10 BKS gap | **Protocol-specific (expensive hybrid)** |
| n40 closeness to BKS | **Uncompetitive (~17%)** |
| Hybrid more efficient than classical search | **Unsupported** without equal-budget evidence |

---

## Regeneration

```bash
python scripts/reporting/scientific_results_audit.py
```
