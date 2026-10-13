# History-Aware Neural Electric Vehicle Routing with DeepSets Fleet Coordination and Exact Nonlinear Charging Optimization

**HAT-EVRP-NL — research manuscript**  
Results index: `results/README.md` · Nonlinear study: `results/NONLINEAR_CHARGING_STUDY.md` · Learning study: `results/REWARD_AND_LEARNING_STUDY.md`  
Tables: `results/tables/` · Figures: `results/figures/` · Raw: `results/raw/`

---

## Abstract

The electric vehicle routing problem with nonlinear charging (E-VRP-NL) couples customer sequencing with station choice and charge quantity under piecewise-linear, state-of-charge–dependent charging times. We study a hybrid architecture—HAT-EVRP-NL—that combines a static node encoder, a causal route-history Transformer, dynamic EV-state fusion, optional DeepSets fleet aggregation, and a hierarchical route/customer pointer, with **exact fixed-route nonlinear charging optimization** via `frvcpy` and independent Python physical validation. Policies are pretrained by behavior cloning of greedy/local-search teachers and may be fine-tuned with REINFORCE; because `frvcpy` is non-differentiable, policy gradients use sampled routing-action log-probabilities and post-optimization rewards.

We organize evidence around four research questions. On matched Cycle-3 fair one-shot local search (n=10, seeds 42–44), **History Transformer Routing** achieves 19.813 h mean full duration versus 19.993 h for a state-only pointer and 20.142 h for randomized greedy+LS; DeepSets does **not** consistently improve upon history alone (HAT-EVRP-NL 19.886 h). A dedicated nonlinear-versus-linear study on all 20 official instances at n=10, 20, and 40 shows that plans optimized under linear charging surrogates frequently become **infeasible under true Montoya physics** (e.g., only 11/20 feasible at n=40 for fixed-route linear plans, versus 20/20 for TRUE-NL). Across three Cycle-3 training seeds, REINFORCE endpoints do **not** consistently beat best behavior-cloning validation checkpoints—especially for the Transformer–DeepSets configuration (0/3 seeds). Contribution: a rigorous experimental evaluation of history-aware hybrid E-VRP-NL routing with explicit charging-physics sensitivity—not a claim of global routing optimality or reliable RL improvement.

---

## 1. Introduction

Battery-electric fleets introduce charging time that is *nonlinear* in remaining capacity: charging slows as the battery approaches full. Planning therefore couples sequencing with station choice and charge quantity. Montoya et al. (2017) formalized this setting as E-VRP-NL and released a benchmark with station-specific piecewise-linear charging curves and a maximum route duration.

The Montoya E-VRP-NL benchmark is particularly appropriate for evaluating the proposed hybrid model because it explicitly represents charging-time nonlinearity through station-specific piecewise-linear curves. Consequently, the charging duration required for a given energy increment depends on the battery state of charge. We therefore examine not only total routing cost, but also the physical feasibility, charging time, and routing decisions obtained when linear charging approximations are used.

Neural construction heuristics—especially attention-based models (Kool et al., 2019)—advance classical VRP by learning sequential decisions. E-VRP-NL adds representation and physics questions that we treat as research questions rather than assumed advantages:

**RQ1.** Does a causal route-history Transformer improve neural EV routing compared with state-only representations?

**RQ2.** Does DeepSets fleet aggregation improve upon the history Transformer?

**RQ3.** Does nonlinear charging optimization matter for feasibility, charging decisions, and objective quality compared with simpler linear charging assumptions?

**RQ4.** Does policy-gradient reinforcement learning actually improve upon behavior cloning?

We evaluate a *hybrid* pipeline: the neural policy proposes customer orders; `frvcpy` solves the fixed-route vehicle charging problem (FRVCP) for each order; a same-project Python forward simulator validates SOC, duration, and coverage. We do **not** claim end-to-end global routing optimality.

---

## 2. Related Work

**Nonlinear E-VRP and linear-assumption sensitivity.** Montoya, Guéret, Mendoza, and Villegas (2017) introduced E-VRP-NL with nonlinear charging, partial recharge, and a 120-instance testbed, and specifically motivate careful treatment of charging nonlinearity. DOI: [10.1016/j.trb.2017.02.004](https://doi.org/10.1016/j.trb.2017.02.004).

**Formulations, FRVCP, and BKS quality.** Froger, Mendoza, Jabali, and Laporte (2019) developed improved MILP formulations and FRVCP algorithms. Froger, Jabali, Mendoza, and Laporte (2022), *Transportation Science* ([10.1287/trsc.2021.1111](https://doi.org/10.1287/trsc.2021.1111)), report improved uncapacitated BKS used here for literature-compatible gap reporting (`results/literature_bks_froger2021.csv`, Table 13 values).

**Attention routing and REINFORCE.** Kool, van Hoof, and Welling (2019) present the Attention Model with REINFORCE and a greedy rollout baseline ([arXiv:1803.08475](https://arxiv.org/abs/1803.08475)). Kwon et al. (2020) propose POMO-style multi-rollout policy-gradient learning ([arXiv:2010.16011](https://arxiv.org/abs/2010.16011)).

**Neural routing plus exact subproblem optimization.** Hybrid pipelines that couple learned construction with exact combinatorial subsolvers are increasingly common; our charging subproblem uses `frvcpy` (Kullman et al., 2021; [10.1287/ijoc.2020.1035](https://doi.org/10.1287/ijoc.2020.1035)), which is optimal for *fixed* customer sequences, not for joint routing.

**Deep Sets.** Zaheer et al. (2017) introduce permutation-invariant set functions \(\rho(\sum_i \phi(x_i))\).

We distinguish established methods from this project’s innovations: causal route-history encoding for E-VRP-NL construction, DeepSets fleet aggregation as an ablation, and an explicit linear-versus-nonlinear charging sensitivity study under true physics.

---

## 3. Problem Formulation

### 3.1 Sets and parameters

- \(N = \{0\} \cup C \cup S\): depot \(0\), customers \(C\), charging stations \(S\).
- Homogeneous EVs with battery capacity \(Q\).
- Travel time \(t_{ij}\) and energy \(e_{ij}\) on arcs.
- Customer service times \(s_i\); maximum route duration \(T_{\max}\) (10 h on official instances).
- Station-specific piecewise-linear concave charging curves; partial recharging allowed.
- Exact customer coverage: each customer visited exactly once.

### 3.2 Charging decisions

For technology \(s\), \(F_s(q)\) is cumulative charging time to energy \(q\). A charge from \(q_\mathrm{in}\) to \(q_\mathrm{out}\) costs \(T_\mathrm{charge}=F_s(q_\mathrm{out})-F_s(q_\mathrm{in})\).

### 3.3 Objectives

Internal full duration (used for training rewards and primary reporting):

\[
Z_\mathrm{full} = T_\mathrm{travel} + T_\mathrm{service} + T_\mathrm{nonlinear\_charge}.
\]

Literature benchmark duration (for BKS gaps):

\[
Z_\mathrm{lit} = T_\mathrm{travel} + T_\mathrm{nonlinear\_charge} = Z_\mathrm{full} - \sum_i s_i.
\]

Service can be subtracted for BKS comparison without changing the ordering of complete feasible solutions that serve the same customer set (fixed total service). **Do not** compare \(Z_\mathrm{full}\) directly with published travel+charging BKS.

---

## 4. Proposed HAT-EVRP-NL Method

Components, in order:

1. **Static node encoder** — shared embeddings of depot, customers, and stations.
2. **Causal route-history Transformer** — encodes the visit sequence of the active EV (ablated off in state-only variants).
3. **Dynamic EV-state fusion** — SOC, elapsed time, location fused with history.
4. **DeepSets fleet representation** — permutation-invariant aggregation over open routes (ablated off when unused).
5. **Hierarchical route/customer pointer** — first select which open route (or open a new one), then select the next customer.
6. **Fixed-route `frvcpy` nonlinear charging optimization** — exact FRVCP given the customer sequence.
7. **Independent Python validation** — SOC, duration, coverage checks on true curves.
8. **Optional classical / neural repair** — local search or learned repair after construction.

**Ablation keys (not separate proposed algorithms):**

| Key | Descriptive name | History | DeepSets |
|---|---|---|---|
| H1 | State-Only Hierarchical Pointer | no | no |
| H2 | History Transformer Routing | yes | no |
| H3 | DeepSets Fleet Routing | no | yes |
| H4 | History Transformer + DeepSets (HAT-EVRP-NL) | yes | yes |

Classical reference: Randomized Greedy Routing (H0) and Randomized Greedy + Local Search (H0_strong).

---

## 5. Training Methodology

The routing policy is first trained through behavior cloning of feasible solutions generated by randomized greedy construction and local search. The expert charging schedules are obtained from an exact fixed-route nonlinear charging solver. The pretrained policy may then undergo REINFORCE fine-tuning, using the negative full route duration as reward for feasible solutions and a fixed penalty for infeasible solutions. Since the fixed-route optimizer is not differentiable, policy gradients are computed from sampled routing-action log probabilities rather than through the charging optimizer.

### 5.1 Behavior cloning

\[
L_\mathrm{BC} = -\sum_t \log \pi_\theta(a_{\mathrm{expert},t}\mid s_t),
\]

with hierarchical route then customer action factorization as implemented in `train.py` / `model.py`.

### 5.2 REINFORCE reward (implemented)

Standard REINFORCE / POMO-style rollouts:

\[
R(\pi)=\begin{cases}-Z_\mathrm{full}(\pi) & \text{if feasible},\\ -100 & \text{if infeasible.}\end{cases}
\]

Repair RL (optional; not Cycle-3 primary): \(R_\mathrm{repair}=Z_\mathrm{before}-Z_\mathrm{after}\) if feasible, else \(-50\).

Policy-gradient loss:

\[
L_\mathrm{RL}=-(R-b)\sum_t\log\pi_\theta(a_t\mid s_t),
\]

with baseline \(b\) = greedy rollout reward (standard REINFORCE) or mean of sampled rollouts (POMO / repair).

**Reward ≠ loss.** Reward is the scalar solution-quality signal; advantage is \(R-b\); REINFORCE loss is a stochastic gradient estimator. BC cross-entropy and policy-gradient losses are not directly comparable. Validation objective is a separate measurement.

### 5.3 Checkpoint selection

At evaluation steps, greedy decode on a held-out **synthetic** validation set scores \((\mathrm{feasible\_rate}, -Z_\mathrm{full})\). The best state is restored before saving `.pt` weights—not necessarily the final RL iterate.

### 5.4 Reward-scale risk

The constant infeasibility penalty \(-100\) is not automatically appropriate at all sizes: if a feasible \(Z_\mathrm{full}>100\), an infeasible rollout rewarded \(-100\) scores *better* than that feasible rollout. We investigate this risk empirically (Section 10) and do **not** silently change historical reward definitions.

Full documentation: `docs/TRAINING_AND_REWARD_METHODOLOGY.md`.

---

## 6. Experimental Setup

| Item | Setting |
|---|---|
| Benchmark | Montoya E-VRP-NL official XML (n=5,10,20,40,80,160 available) |
| Primary architecture eval | n=10, 20 official instances, seeds 42–44 (Cycle-3 fair-LS) |
| Scaling / hybrid | n=20,40 under documented protocols; n=40 recovered hybrid is single-seed |
| Nonlinear vs linear (RQ3) | **All 20** official instances at n=10,20,40 |
| Training | Synthetic generator v2; 120 train / 12 val; BC 400 + RL 40 (Cycle-3) |
| Charging optimizer | `frvcpy` FixedRouteOracle |
| Hardware | CPU Python experiments (see logs) |
| Metrics | Feasibility, \(Z_\mathrm{full}\), \(Z_\mathrm{lit}\), charging hours, oracle counts, runtime |
| Objective for BKS | \(Z_\mathrm{lit}\) only |

Unequal-compute protocols are labeled and never mixed with matched fair-LS claims.

---

## 7. Results — Nonlinear Charging (RQ3)

*Major section.* Full study: `results/NONLINEAR_CHARGING_STUDY.md`. Table: `table15_nonlinear_vs_linear_charging.md`. Figures: `fig_main02_*`–`fig_main04_*`.

### 7.1 Charging curves and physical interpretation

Station technologies (slow / normal / fast) have distinct concave PWL curves. **LINEAR-SECANT** uses the empty→full chord (average rate). **LINEAR-EARLY-RATE** extrapolates the first segment (optimistic on concave curves). Figure 2 compares curves on a representative instance.

### 7.2 Linear versus nonlinear planning assumptions

On **matched customer sequences** (fixed-route protocol):

| n | TRUE-NL feasible (true physics) | LINEAR-SECANT | LINEAR-EARLY-RATE |
|---:|---:|---:|---:|
| 10 | 20/20 | 16/20 | 16/20 |
| 20 | 20/20 | 12/20 | 12/20 |
| 40 | 20/20 | 11/20 | 11/20 |

TRUE-NL re-optimization after a linear plan (`repair_TRUE-NL_after_*`) restores 20/20 feasibility and is reported as a **separate** repaired method.

### 7.3 Feasibility under true charging physics

**LINEAR-EARLY-RATE** often plans successfully under the surrogate but fails true-physics duration checks (e.g., 9 duration violations at n=40 fixed-route). **LINEAR-SECANT** frequently fails already under the planner. End-to-end routing with frozen H2 weights shows the same qualitative pattern: at n=40, TRUE-NL 20/20 vs LINEAR-SECANT 11/20 vs LINEAR-EARLY-RATE 9/20 under true physics.

### 7.4 Charging duration and SOC behavior

Among instances where both TRUE-NL and a linear plan remain true-feasible, paired true charging-time deltas are reported in `fig_main03b_*`. Feasible-only means must not be used to rank methods when feasibility rates differ.

### 7.5 Impact on routing objective and decisions

End-to-end fingerprints (`end_to_end_decision_changes.csv`) show that the assumed charging law changes route sequences and charger usage. Objective decomposition under TRUE-NL fixed-route plans separates travel, service, and nonlinear charging (`fig_main04_*`).

**Answer to RQ3.** Nonlinear charging matters in practice: linear surrogates produce plans that are frequently infeasible (or planner-rejected) when evaluated under true Montoya physics, with the failure rate growing with instance size. This is a feasibility result first, not a mean-duration ranking among survivors.

**Limitation.** Fixed-route sequences come from one H2+LS seed; surrogates are two explicit linear models, not every approximation in the literature.

---

## 8. Results — Neural Routing Quality (RQ1, RQ2)

### 8.1 State-only vs history Transformer (RQ1)

**Protocol:** Cycle-3 fair best-of-10 constructions + one-shot basic LS; n=10; seeds 42–44; full duration.

| Method | Mean full h |
|---|---:|
| Randomized Greedy + LS | 20.142 |
| State-Only Hierarchical Pointer (H1) | 19.993 |
| History Transformer Routing (H2) | **19.813** |
| DeepSets Fleet Routing (H3) | 20.109 |
| History Transformer + DeepSets (H4) | 19.886 |

History improves under this protocol (H2 vs H1: −0.180 h). **Limitation:** gains are measured under hybrid FRVCP evaluation, not pure neural charging.

### 8.2 DeepSets incremental contribution (RQ2)

On the same matched n=10 fair-LS protocol, H4 (19.886 h) does **not** improve on H2 (19.813 h). Win/tie/loss and CI details are in `results/tables/table06_h2_vs_h4.md` / Cycle-3 reports.

**Recovered n=40 exploratory hybrid** (20 official instances, one seed, **unequal** realized search compute): History Transformer 56.806 h; History Transformer + DeepSets 56.370 h. H4 is better on 7 instances, tied on 4, worse on 9. The mean advantage does **not** justify declaring reliable DeepSets superiority.

Objective decomposition (same FRVCP for both):

| Variant | Travel | Service | NL charge | Total |
|---|---:|---:|---:|---:|
| History Transformer | 33.745 | 20.000 | 3.060 | 56.806 |
| History Transformer + DeepSets | 33.219 | 20.000 | 3.151 | 56.370 |

For the 40-customer recovered-hybrid experiment, both neural variants are evaluated using the same exact fixed-route nonlinear charging procedure. The Transformer–DeepSets configuration achieves a lower mean full duration than the history-only Transformer, predominantly through reduced travel time, despite slightly greater charging duration. Hence this particular result does not demonstrate superior charging optimization by the DeepSets architecture.

### 8.3 Classical baseline comparison

Under fair-LS, neural history-aware construction improves on randomized greedy+LS on n=10 (above). Scaling tables: `table05_scalability.md`.

### 8.4–8.6 Scaling, compute, and search

Post-search gains must not be attributed to architecture alone. Direct learned charging (no `frvcpy`) achieves feasible complete solutions on only **4/20** official n=10 cases versus **20/20** for the frvcpy hybrid in that experiment—supporting an exact charging component, but **not** substituting for the linear-versus-nonlinear study in Section 7.

Corrected literature gaps use \(Z_\mathrm{lit}\); see `table08_bks_comparison.md`.

---

## 9. Results — Learning and Reward (RQ4)

Full study: `results/REWARD_AND_LEARNING_STUDY.md`.

### 9.1 Behavior-cloning learning curves

Cycle-3 BC reduces validation duration over 400 steps (see `fig12_*`). Expert actions come from greedy/LS teachers with frvcpy schedules.

### 9.2 REINFORCE reward dynamics

Historical Cycle-3 CSVs lack per-step sampled reward trajectories. The controlled RQ4 run instruments reward, greedy baseline, advantage, feasibility, PG loss, entropy, and grad norm (`results/raw/reward_learning/reward_*.csv`). Publication figure: `fig_main07_reward_dynamics.png`. The former placeholder `fig14_reward_dynamics_unavailable.png` is archived and is **not** a publication figure.

### 9.3 BC-only versus RL fine-tuning

**A. Controlled RQ4** (fresh BC-only → 40 standard REINFORCE steps; official n=10 **construction-only**; seeds 42–44):

| Model | Seeds where RL mean < BC mean | Mean BC (h) | Mean RL (h) |
|---|---|---:|---:|
| History Transformer (H2) | **1/3** | 22.157 | 22.880 |
| History + DeepSets (H4) | **3/3** | 22.477 | 21.636 |

All evaluations 20/20 feasible. These are **not** Cycle-3 fair-LS numbers.

**B. Historical Cycle-3 log analysis** (synthetic validation endpoints; not the controlled bake-off):

| Model | Mean best-BC val (h) | Mean RL endpoint (h) | Seeds RL end < best BC |
|---|---:|---:|---:|
| H2 | 15.4235 | 22.0845 | **1/3** |
| H4 | 15.4325 | 26.4013 | **0/3** |

Across the three Cycle-3 training seeds, reinforcement-learning endpoints do not demonstrate consistent improvement over the best behavior-cloning validation checkpoints. This effect is especially evident in the Transformer–DeepSets configuration, for which all three RL endpoints are worse than the corresponding best BC validation result. The deployed checkpoints are selected by synthetic-validation feasibility and duration, preventing poor RL endpoints from automatically replacing better pretrained weights.

Distinguish: last BC step · best selected BC step · final RL step · best deployed checkpoint.

### 9.4 Training stability and infeasibility

Reward CSVs show large intermittent gradient norms. The fixed \(-100\) infeasibility penalty is a known scale risk when \(Z_\mathrm{full}\) can exceed 100. Alternative scalings require new controlled experiments.

### 9.5 Checkpoint-selection outcomes

Deployed Cycle-3 `.pt` files reflect validation-selected weights, which may coincide with BC-best rather than RL-end when RL degrades on synthetic validation.

**Answer to RQ4.** Evidence is **protocol-dependent**: controlled short RL from fresh BC improves official construction-only means for H4 (3/3 seeds) but not H2 (1/3); historical Cycle-3 synthetic-val RL endpoints are often worse than best BC (H4: 0/3). We do **not** claim that RL reliably improves deployed Cycle-3 H4 checkpoints.

**Limitation.** Controlled RQ4 uses construction-only evaluation and a short RL budget; Cycle-3 fair-LS multi-seed quality remains the primary routing-quality protocol for RQ1–RQ2.

---

## 10. Ablations and Limitations

| Topic | Finding / limit |
|---|---|
| History vs state-only | History helps on matched n=10 fair-LS |
| DeepSets | No consistent gain over history on matched n=10; n=40 hybrid inconclusive under unequal compute |
| Exact vs learned charging | frvcpy hybrid 20/20 vs direct CHARGE 4/20 on tested n=10 |
| Nonlinear vs linear | Linear plans often infeasible under true physics (Section 7) |
| Classical search vs neural repair | Post-search gains ≠ architecture gains |
| BC vs RL | RL endpoints often worse; selection protects BC-best |
| Runtime / oracle cost | Documented per experiment; do not mix unequal budgets |
| Size generalization | Strongest multi-seed claims on n=10; larger sizes have thinner seed coverage |
| Statistical uncertainty | Three seeds; report win/tie/loss where available |

---

## 11. Conclusion

| RQ | Answer (evidence-based) |
|---|---|
| RQ1 | Yes under Cycle-3 fair-LS n=10: history Transformer beats state-only (19.813 vs 19.993 h). |
| RQ2 | No consistent DeepSets gain over history on matched n=10; n=40 hybrid mean edge is instance-unstable and unequal-compute. |
| RQ3 | Yes—nonlinear charging matters: linear surrogates yield substantial true-physics infeasibility (e.g., 9/20–11/20 failures at n=40). |
| RQ4 | Protocol-dependent: Cycle-3 synthetic-val RL endpoints often worse than best BC (H4 0/3); controlled short RL from fresh BC improves official construction-only H4 (3/3) but not H2 (1/3). |

We do not claim end-to-end global optimality, reliable H4 dominance over H2, or that RL consistently improves deployed Cycle-3 H4 checkpoints.

---

## References (selected)

1. Montoya et al. (2017). *Transportation Research Part B*. https://doi.org/10.1016/j.trb.2017.02.004  
2. Froger et al. (2022). *Transportation Science*. https://doi.org/10.1287/trsc.2021.1111  
3. Kool et al. (2019). Attention Model. https://arxiv.org/abs/1803.08475  
4. Kwon et al. (2020). POMO. https://arxiv.org/abs/2010.16011  
5. Kullman et al. (2021). `frvcpy`. https://doi.org/10.1287/ijoc.2020.1035  

---

## Main figures and tables (publication set)

| Fig | File | Content |
|---|---|---|
| 1 | `fig11_training_supervision_flow.png` / architecture figs | Architecture and training/inference paths |
| 2 | `fig_main02_nonlinear_vs_linear_curves.png` | Nonlinear vs linear charging curves |
| 3 | `fig_main03_linear_plan_true_feasibility.png` (+ e2e/paired) | Feasibility under true physics |
| 4 | `fig_main04_objective_decomposition_nl.png` | Travel + service + NL charging |
| 5 | Matched neural comparison (Cycle-3 figures) | Descriptive method names |
| 6 | `fig12_*` learning curves | BC and RL separated |
| 7 | `fig_main07_reward_dynamics.png` | Real reward logs (when available) |
| 8 | Quality vs compute figures | Budgeted search |

| Table | File |
|---|---|
| 1 | `table01_benchmark_overview.md` |
| 2 | `table10_training_configuration.md` |
| 3 | `table03_main_models.md` |
| 4 | `table15_nonlinear_vs_linear_charging.md` |
| 5 | `table02_objective_and_charging.md` |
| 6 | `table06_bc_vs_rl.md` / `table12_bc_vs_rl.md` |
| 7 | Architecture/search ablations |
| 8 | Runtime / oracle counts |

Secondary plots remain under `results/figures/` or archive; do not treat chronological experiment dumps as main-paper evidence.
