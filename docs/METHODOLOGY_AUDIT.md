# Methodology Audit — HAT-EVRP-NL (RQ1–RQ4)

Status after Phase-1 correctness fixes. This document records **strengths, confounders, and required controlled experiments**. It does not invent new numerical results.

## Shared conventions

| Symbol | Meaning |
|---|---|
| \(Z_\mathrm{full}\) | Travel + service + nonlinear charging (hours) |
| \(Z_\mathrm{lit}\) | \(Z_\mathrm{full} - \sum_i T_{\mathrm{service},i}\) |
| BKS gap | \(100(Z_\mathrm{lit}-\mathrm{BKS})/\mathrm{BKS}\) (Froger Table 13) |
| Oracle executions | Actual `frvcpy` solver runs (cache misses) |
| Oracle requests | Logical `solve()` calls including cache hits |

Never compare \(Z_\mathrm{full}\) directly to literature BKS.

---

## RQ1 — History Transformer

**Current evidence:** Modest gain of H2 over H1 on matched n=10 Cycle-3 fair-LS.

**Strengths**
- Causal Transformer over route history is a clear architectural hypothesis.
- Matched fair-LS protocol isolates architecture from hybrid search.

**Confounders / risks**
- History may include information derived from `frvcpy`-closed plans (terminal depot / charge structure) even when the decoder constructs open routes. If so, part of the “history” signal is oracle-dependent, not pure causal prefix structure.
- Teacher quality and BC multi-target settings can interact with history usefulness.

**Required controlled ablations (do not change the default architecture until run)**
1. H1 state-only (existing).
2. H2 optimized-plan history (existing default).
3. **Prefix-only history:** encode only decisions already committed by the decoder (no post-hoc frvcpy plan features).
4. Matched seeds, data, decoder, and search budgets.

**Reporting rule:** Attribute gains to history only under (2) vs (1) and (3) vs (1) with layout-aware paired CIs.

---

## RQ2 — DeepSets fleet coordination

**Current evidence:** H4 does **not** consistently beat H2 (n10 fair-LS layout CI includes zero; n40 hybrid mean lower but unpaired wins/ties/losses 7/4/9 with CI crossing zero).

**Likely reasons to investigate**
- Sum pooling without route-count normalization can confound fleet size.
- Aggregating finished routes may dilute open-route coordination.
- Single-target BC may under-supervise fleet-level decisions.
- Any H4 edge on n40 may be travel-sequence luck, not DeepSets (same `frvcpy`).

**Controlled ablations (matched)**
- Sum vs mean pooling (+ route-count features).
- All-route vs open-route-only aggregation (`H4_open` already exists).
- Single-target vs multi-target BC vs fleet-counterfactual supervision.
- Identical data, steps, seeds, decoder, and search.

**Reporting rule:** If no reliable paired layout advantage remains, state the **negative finding** explicitly. Do not add complexity without a demonstrated need.

---

## RQ3 — Nonlinear charging

**Current evidence:** Strong. Linear SECANT / EARLY-RATE harm true-physics feasibility at n=40 (fixed-route and end-to-end).

**Strengths**
- TRUE-NL vs LINEAR-SECANT vs LINEAR-EARLY-RATE are separate planners.
- True Montoya validation is independent of the planner.
- Planner rejection ≠ false feasibility.

**Remaining causal gaps**
1. Changing curves affects both the oracle **and** neural features if features encode charge rates. Need an experiment that freezes network inputs while swapping oracle physics (and the reverse).
2. Route-change rates must use **vehicle-order-invariant** fingerprints (now implemented: `route_sequence_fingerprint`). Recompute decision-change percentages after regenerating e2e fingerprints.
3. Feasible-only mean objectives must not be ranked across models with different feasibility rates.
4. TRUE-NL repair after a linear plan is an **additional algorithm**, not a property of the linear model.

**Contribution framing:** Hybrid neural routing + exact fixed-route FRVCP-NL evaluation. `frvcpy` itself is an existing algorithmic component, not a novel charging optimizer claimed by this paper.

---

## RQ4 — BC vs REINFORCE

**Current evidence:** Controlled official n10 construction-only: H2 improves 1/3 seeds; H4 improves 3/3. Historical Cycle-3 synth-val RL often worsens H4 vs best BC.

**Reward issues to audit**
\[
R=-Z_\mathrm{full}\ \text{if feasible},\quad R=-100\ \text{otherwise}.
\]
- Feasible routes with \(Z_\mathrm{full}>100\) receive **worse** reward than the infeasible penalty (−100). Check whether such long feasible routes occur; if so, rescale or use a soft feasibility barrier.
- Validation-selected checkpoints ≠ last RL step ≠ BC-only.

**Required protocol**
- Identical BC init weights for paired BC-only vs RL-from-BC.
- Separate reporting of BC-only / last-RL / val-selected.
- Matching instances and search mode.
- ≥3 (preferably ≥5) training seeds; layout-aware paired stats.
- **Do not mix** Cycle-3 historical RL with controlled RQ4.
- **Do not claim** RL improves deployed hybrid unless matched hybrid evaluations support it.

---

## Cross-cutting experimental risks

| Risk | Mitigation |
|---|---|
| Equal eval-count ≠ equal wall-clock | Report both; enforce real deadlines when claiming wall-clock fairness |
| Architecture vs search confounding | Freeze search when comparing H1–H4 |
| Post-hoc tuning on test | Freeze configs before confirmatory runs |
| Seed×instance inflation of N | Pair on layouts; report seed variability separately |
| Near-zero n10 hybrid BKS gap | Disclose expensive protocol; never equate to fair-LS ~5% |

---

## Priority order for new science (after Phase 1)

1. Prefix-only history ablation (RQ1) — clarifies whether history is causal. **NOT YET TRAINED** (manifest).  
2. Recompute e2e route-change rates with invariant fingerprints (RQ3). **DONE** (versioned reanalysis).  
3. DeepSets pooling / open-only matched ablations (RQ2) — expect possible negative result. Lightweight P0 shows H2=H4; H4_open unmatched. **Matched multi-seed NOT YET RUN.**  
4. Reward scaling audit + controlled RL with frozen search (RQ4). Construction-only exists; **hybrid RL NOT YET RUN.**  
5. Only then: multi-seed confirmatory n20/n40 sweeps.

Strict wall-clock LS deadlines are implemented (`local_search(deadline=...)`). Full P3 tables remain to be generated under authorization.
