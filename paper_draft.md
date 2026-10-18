# History-Aware Neural Electric Vehicle Routing with DeepSets Fleet Coordination and Exact Nonlinear Charging Optimization

**HAT-EVRP-NL — research manuscript**  
Results: `results/README.md` · Nonlinear: `results/NONLINEAR_CHARGING_STUDY.md` · Learning: `results/REWARD_AND_LEARNING_STUDY.md`  
Figures: `results/figures/` (20 main) + 1 supplementary diagnostic · Catalog: `docs/FIGURE_CATALOG.md` · Tables: `results/tables/` · Raw: `results/raw/`

---

## Abstract

The electric vehicle routing problem with nonlinear charging (E-VRP-NL) couples customer sequencing with station choice and charge quantity under piecewise-linear, state-of-charge–dependent charging times. We study a hybrid architecture—HAT-EVRP-NL—that combines a static node encoder, a causal route-history Transformer, dynamic EV-state fusion, optional DeepSets fleet aggregation, and a hierarchical route/customer pointer, with **exact fixed-route nonlinear charging optimization** via `frvcpy` and independent Python physical validation. Policies are pretrained by behavior cloning and may be fine-tuned with REINFORCE; because `frvcpy` is non-differentiable, policy gradients use sampled routing-action log-probabilities and post-optimization rewards.

On matched Cycle-3 fair one-shot local search (n=10, seeds 42–44), Route Transformer (RT) achieves 19.813 h mean full duration versus 19.993 h for a State Pointer and 20.142 h for randomized greedy+LS; DeepSets does **not** consistently improve upon history alone (19.886 h). The History versus State-only gain is **seed-sensitive** (≈ −0.58 / +0.04 / ≈0 h on seeds 42–44). A nonlinear-versus-linear study on all 20 official instances at n=10, 20, and 40 shows that linear charging surrogates frequently become **infeasible under true Montoya physics** (e.g., 11/20 at n=40 fixed-route vs 20/20 TRUE-NL). Controlled BC→RL on official n=10 construction-only improves RT + DeepSets on 3/3 seeds but Route Transformer (RT) on only 1/3; historical Cycle-3 synthetic-validation RL endpoints were often worse than best BC. Contribution: a rigorous experimental evaluation—not a claim of global routing optimality or reliable RL improvement for all architectures.

---

## 1. Introduction

Battery-electric fleets introduce charging time that is *nonlinear* in remaining capacity. Montoya et al. (2017) formalized E-VRP-NL with station-specific piecewise-linear charging curves and a maximum route duration.

The Montoya E-VRP-NL benchmark is particularly appropriate for evaluating the proposed hybrid model because it explicitly represents charging-time nonlinearity through station-specific piecewise-linear curves. Consequently, the charging duration required for a given energy increment depends on the battery state of charge. We therefore examine not only total routing cost, but also the physical feasibility, charging time, and routing decisions obtained when linear charging approximations are used.

**RQ1.** Does a causal route-history Transformer improve neural EV routing compared with state-only representations?  
**RQ2.** Does DeepSets fleet aggregation improve upon the history Transformer?  
**RQ3.** Does nonlinear charging optimization matter versus linear charging assumptions?  
**RQ4.** Does policy-gradient RL improve upon behavior cloning?

---

## 2. Related Work

Montoya et al. (2017) [10.1016/j.trb.2017.02.004](https://doi.org/10.1016/j.trb.2017.02.004). Froger et al. (2022) BKS [10.1287/trsc.2021.1111](https://doi.org/10.1287/trsc.2021.1111). Kool et al. (2019) Attention Model [arXiv:1803.08475](https://arxiv.org/abs/1803.08475). Kwon et al. (2020) POMO [arXiv:2010.16011](https://arxiv.org/abs/2010.16011). `frvcpy` (Kullman et al., 2021) [10.1287/ijoc.2020.1035](https://doi.org/10.1287/ijoc.2020.1035).

---

## 3. Problem Formulation

Graph \(N=\{0\}\cup C\cup S\), battery \(Q\), travel times/energies, service times, \(T_{\max}\), station-specific PWL charging \(F_s(q)\), partial recharge, exact coverage.

\[
Z_\mathrm{full}=T_\mathrm{travel}+T_\mathrm{service}+T_\mathrm{nonlinear\_charge},\quad
Z_\mathrm{lit}=T_\mathrm{travel}+T_\mathrm{nonlinear\_charge}.
\]

Service can be subtracted for BKS comparison without changing the ordering of complete feasible solutions serving the same customers.

---

## 4. Proposed HAT-EVRP-NL Method

![Figure 1. HAT-EVRP-NL architecture](results/figures/fig01_hat_evrp_nl_architecture.png)

**Figure 1.** Hybrid HAT-EVRP-NL architecture. Blue blocks are learned (node encoder, causal history Transformer, dynamic state fusion, optional DeepSets, hierarchical pointer). Orange blocks are algorithmic and non-differentiable (`frvcpy` FRVCP-NL and Python physical checker). Green blocks denote data/outputs. Training: synthetic experts → BC → optional REINFORCE → validation-selected checkpoint. Gradients never pass through `frvcpy`. Ablations compare State Pointer, Route Transformer (RT), Fleet DeepSets, and Route Transformer + DeepSets. Policy gradients use sampled routing log-probabilities; charging is optimized post-decode.

Ablations (display names): State Pointer · Route Transformer (RT) · Fleet DeepSets · Route Transformer + DeepSets (HAT-EVRP-NL). Classical: Greedy Construction / Greedy + Local Search. Internal checkpoint keys H0–H4 are retained for reproducibility only (`presentation_names.py`).

![Figure 2. Montoya layouts](results/figures/fig02_montoya_instance_layouts.png)

**Figure 2.** Official Montoya spatial layouts for location families tc0, tc1, and tc2 (n=20 representatives). Markers: depot (square), customers (circles), charging stations by technology (triangles). Coordinates are dataset units from the XML files.

![Figure 3. Feasible EV routes](results/figures/fig03_feasible_ev_routes.png)

**Figure 3.** Validated hybrid solution for `tc2c40s5cf3` (Route Transformer (RT), hybrid_insert; full duration 40.14 h). Routes from saved JSON; open circles mark charge visits. Source: `results/raw/regression_recovery/solutions/`.

---

## 5. Training Methodology

The routing policy is first trained through behavior cloning of feasible solutions generated by randomized greedy construction and local search. Expert charging schedules come from an exact fixed-route nonlinear charging solver. The pretrained policy may then undergo REINFORCE fine-tuning with \(R=-Z_\mathrm{full}\) if feasible and \(R=-100\) if infeasible. Because the fixed-route optimizer is not differentiable, policy gradients use sampled routing-action log probabilities rather than gradients through `frvcpy`.

Details: `docs/TRAINING_AND_REWARD_METHODOLOGY.md`.

---

## 6. Experimental Setup

Official Montoya XML; primary multi-seed architecture eval on n=10 fair-LS (seeds 42–44); nonlinear study covers all 20 instances at n=10/20/40; recovered n=40 hybrid is single-seed; training on synthetic n=10. Metrics: feasibility, \(Z_\mathrm{full}\), \(Z_\mathrm{lit}\), charging hours, oracle counts, runtime.

---

## 7. Results — Nonlinear Charging (RQ3)

Full study: `results/NONLINEAR_CHARGING_STUDY.md`.

![Figure 4. Nonlinear charging curves](results/figures/fig04_nonlinear_charging_curves.png)

**Figure 4.** Montoya piecewise-linear charging curves versus LINEAR-SECANT and LINEAR-EARLY-RATE for slow/normal/fast technologies. Late-SOC region charging slows relative to early-rate extrapolation. Units: hours vs kWh. XML breakpoints preserved.

![Figure 5. Marginal charging time vs SOC](results/figures/fig05_charging_time_vs_soc.png)

**Figure 5.** Marginal charging time (minutes per kWh) from actual PWL segment slopes versus SOC fraction. Higher SOC requires more time for the same energy increment—the physical reason linear early-rate models are optimistic.

![Figure 6. Fixed-route feasibility](results/figures/fig06_fixed_route_charging_feasibility.png)

**Figure 6.** Fixed-route charging optimization on matched customer sequences (20 official instances per size; separate panels). Stacked outcomes: true-physics feasible (green), planner-OK but true-physics fail (orange), planner rejected (grey). Numbers = true-feasible counts. At n=40: TRUE-NL 20/20; SECANT 11/20; EARLY 11/20—with different failure mechanisms.

![Figure 7. End-to-end feasibility](results/figures/fig07_end_to_end_charging_feasibility.png)

**Figure 7.** End-to-end neural routing with frozen Route Transformer (RT) weights (20 official instances per size). Left three panels: true-physics feasibility under each planning curve (TRUE-NL / Secant / Early). Right panel: fraction of instances whose customer sequences change relative to TRUE-NL. Unlike Figure 6, routing decisions may change when the planner uses a misspecified charging law. At n=40: TRUE-NL 20/20, SECANT 11/20, EARLY 9/20.

![Figure 8. SOC charging schedule](results/figures/fig08_soc_charging_schedule.png)

**Figure 8.** Same fixed customer sequence on `tc0c10s2cf1` (route with charges): TRUE-NL plan (left) versus LINEAR-EARLY-RATE plan replayed on true curves (right). Shaded intervals = charging; dashed line = \(T_{\max}\). The linear plan underestimates charging duration and violates the route-duration limit. Companion table: `results/tables/table_fig08_schedule_case.md`.

![Figure 9. Charging-time estimation error](results/figures/fig09_linear_charging_estimation_error.png)

**Figure 9.** Planner-predicted charging hours minus true nonlinear charging hours for fixed-route plans with both quantities recorded. Negative values = underestimation. Sample sizes shown per panel (selected by recorded hours—not an unconditional sample of all 20).

![Figure 10. Station usage](results/figures/fig10_station_usage_and_energy.png)

**Figure 10.** Recovered n=40 hybrid_insert (20 instances): (A) charger visits by technology, (B) mean energy charged, (C) mean nonlinear charging duration—**separate panels, no twin-axis**. Describes what `frvcpy` chooses under TRUE-NL for Route Transformer (RT) vs RT + DeepSets.

![Figure 11. Objective decomposition](results/figures/fig11_objective_decomposition.png)

**Figure 11.** Mean travel + service + nonlinear charging for recovered n=40 hybrid_insert (20 instances). Route Transformer (RT): 33.745 + 20.000 + 3.060 = 56.806 h; RT + DeepSets: 33.219 + 20.000 + 3.151 = 56.370 h. Legend outside axes. The mean edge is mainly travel, not charging—both use the same `frvcpy` optimizer (not evidence that DeepSets improves the charging solver).

**Answer to RQ3.** Nonlinear charging matters: linear surrogates frequently yield planner failures or true-physics infeasibility, with rates growing in \(n\).

---

## 8. Results — Neural Routing Quality (RQ1, RQ2)

![Figure 12. Architecture ablation](results/figures/fig12_neural_architecture_ablation.png)

**Figure 12.** Matched n=10 Cycle-3 fair one-shot LS (20 instances × 3 seeds). **A:** seed means ± std across seeds. **B:** per-seed paired means showing RT − State Pointer is seed-sensitive (≈ −0.58 / +0.04 / ≈0 h). Route Transformer (RT) 19.813 h average beats State Pointer 19.993 h and Greedy+LS 20.142 h; RT + DeepSets 19.886 h does not consistently beat RT alone.

![Figure 13. Paired effects](results/figures/fig13_history_deepsets_paired_effect.png)

**Figure 13.** Layout-aware paired effects (n=10 Cycle-3 fair-LS): each point is one customer layout after averaging seed-level deltas; diamonds are mean ± 1.96·SE over **n=20 layouts**. Negative = second method better. Do not treat 60 instance×seed rows as 60 independent layouts. RT+DeepSets − RT straddles zero—DeepSets is not a dependable paired win here.

![Figure 14. Literature BKS gaps](results/figures/fig14_literature_bks_gap.png)

**Figure 14.** Literature BKS gaps with \(Z_\mathrm{lit}=Z_\mathrm{full}-\text{service}\) (Froger Table 13), **protocol-separated**: (A) n=10 Cycle-3 fair-LS ≈ 4.99% RT / 5.44% RT+DeepSets; (B) expensive n=10 hybrid insert ≈ **0.027%** (near-BKS; different compute regime—panel B uses a much smaller vertical scale than A/C and is not zero); (C) n=40 recovered hybrid ≈ **17.72% / 16.79%**. Panels are not interchangeable rankings. We do **not** claim state-of-the-art performance.

![Figure 15. Generalization](results/figures/fig15_generalization_scalability.png)

**Figure 15.** Size transfer of n10-trained Route Transformer (RT) and RT + DeepSets under fair one-shot LS: (A) absolute mean \(Z_\mathrm{full}\), (B) architecture difference Δ = (RT+DeepSets) − RT, (C) feasibility rate. Absolute curves largely overlap. The n=80 point uses only 3 instances and is exploratory—not a strong large-instance generalization claim. Recovered hybrid n=40 (different protocol) reaches 56.806 / 56.370 h.

![Figure 16. Search stages](results/figures/fig16_search_improvement_stages.png)

**Figure 16.** Recovered n=40 search-mode comparison as a categorical dot plot (Route Transformer (RT) vs Route Transformer + DeepSets; 20 instances; mean \(Z_\mathrm{full}\), lower better). Modes are independent alternatives from the same starts—not a sequential optimization trajectory. Strong LS need not dominate Basic LS. Hybrid insert has the best mean among reported modes; this is not an equal-budget efficiency claim.

![Figure 17. Quality vs compute](results/figures/fig17_quality_vs_compute.png)

**Figure 17.** Exploratory: Route Transformer (RT) n=40 quality versus stage-level `frvcpy` oracle executions (20 instances, seed 42). Legend outside plot; runtimes where measured. **Unequal realized budgets—not an equal-budget efficiency experiment.**

---

## 9. Results — Learning and Reward (RQ4)

![Figure 18. BC learning curves](results/figures/fig18_behavior_cloning_learning_curves.png)

**Figure 18.** Cycle-3 behavior-cloning synthetic-validation objective in **separate architecture panels** at logged evaluation steps (not continuous interpolation). Bold ≈ seed 42; faint = 43/44; ★ = best logged point per seed.

![Figure 19. Controlled BC vs RL](results/figures/fig19_bc_vs_rl_checkpoint_quality.png)

**Figure 19.** Controlled RQ4 paired slopes: BC-only → RL-from-BC mean full duration on official n=10 **construction-only** (20/20 feasible). Route Transformer (RT): RL better on 1/3 seeds; RT+DeepSets: 3/3. This is **not** proof that RL improves final hybrid routing. Distinct from historical Cycle-3 synthetic-validation RL endpoints (RT+DeepSets often worse than best BC). RT tokens are optimizer-enriched stop representations (including charging features), not a claim that every token is a committed historical action.

![Figure 20. Reward dynamics](results/figures/fig20_reinforce_reward_dynamics.png)

**Figure 20.** Actual REINFORCE rewards from `reward_*_rl_from_bc_rq4.csv`. \(R=-Z_\mathrm{full}\) if feasible, else \(-100\). Faint = seeds; bold = mean rolling median (window=5); dashed = greedy baseline. **Noisy traces do not by themselves demonstrate consistent RL learning**—interpret with Figure 19 checkpoint outcomes and validation-based selection.

Supplementary diagnostic (not a main-paper claim figure): `results/supplementary/figures/fig21_rl_training_stability.png` (gradient norm, advantage, entropy, PG loss; correlational only).

Across Cycle-3 training seeds, RL endpoints do not consistently improve over best BC validation—especially History+DeepSets (0/3). Deployed checkpoints use validation selection.

---

## 10. Ablations and Limitations

History helps on matched n=10 fair-LS; DeepSets lacks consistent gain; linear charging assumptions harm true-physics feasibility; frvcpy hybrid ≫ direct learned charging (4/20 vs 20/20 on tested n=10); post-search ≠ architecture; RL gains are protocol-dependent; n=40 hybrid is single-seed / unequal compute; n=80 thin.

---

## 11. Conclusion

| RQ | Answer |
|---|---|
| RQ1 | Yes under Cycle-3 fair-LS n=10 (history vs state-only). |
| RQ2 | No consistent DeepSets gain over history on matched n=10. |
| RQ3 | Yes—linear surrogates often fail true Montoya physics. |
| RQ4 | Protocol-dependent; do not claim RL always improves deployed Cycle-3 History+DeepSets. |

---

## Manuscript figure index (20 main + 1 supplementary diagnostic)

| Fig | File |
|---|---|
| 1 | `results/figures/fig01_hat_evrp_nl_architecture.png` |
| 2 | `results/figures/fig02_montoya_instance_layouts.png` |
| 3 | `results/figures/fig03_feasible_ev_routes.png` |
| 4 | `results/figures/fig04_nonlinear_charging_curves.png` |
| 5 | `results/figures/fig05_charging_time_vs_soc.png` |
| 6 | `results/figures/fig06_fixed_route_charging_feasibility.png` |
| 7 | `results/figures/fig07_end_to_end_charging_feasibility.png` |
| 8 | `results/figures/fig08_soc_charging_schedule.png` |
| 9 | `results/figures/fig09_linear_charging_estimation_error.png` |
| 10 | `results/figures/fig10_station_usage_and_energy.png` |
| 11 | `results/figures/fig11_objective_decomposition.png` |
| 12 | `results/figures/fig12_neural_architecture_ablation.png` |
| 13 | `results/figures/fig13_history_deepsets_paired_effect.png` |
| 14 | `results/figures/fig14_literature_bks_gap.png` |
| 15 | `results/figures/fig15_generalization_scalability.png` |
| 16 | `results/figures/fig16_search_improvement_stages.png` |
| 17 | `results/figures/fig17_quality_vs_compute.png` |
| 18 | `results/figures/fig18_behavior_cloning_learning_curves.png` |
| 19 | `results/figures/fig19_bc_vs_rl_checkpoint_quality.png` |
| 20 | `results/figures/fig20_reinforce_reward_dynamics.png` |
| Supp. | `results/supplementary/figures/fig21_rl_training_stability.png` |

Catalog: `docs/FIGURE_CATALOG.md`. Supplementary: `results/supplementary/figures/`.

---

## References (selected)

1. Montoya et al. (2017). https://doi.org/10.1016/j.trb.2017.02.004  
2. Froger et al. (2022). https://doi.org/10.1287/trsc.2021.1111  
3. Kool et al. (2019). https://arxiv.org/abs/1803.08475  
4. Kwon et al. (2020). https://arxiv.org/abs/2010.16011  
5. Kullman et al. (2021). https://doi.org/10.1287/ijoc.2020.1035  
6. Chung, Lee, and Tsang (2025). Neural combinatorial optimization with reinforcement learning in industrial engineering: a survey. https://doi.org/10.1007/s10462-024-11045-1  
