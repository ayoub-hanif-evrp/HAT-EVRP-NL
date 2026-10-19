# History-Aware Neural Electric Vehicle Routing with DeepSets Fleet Coordination and Exact Nonlinear Charging Optimization

**HAT-EVRP-NL — research manuscript**  
Results: `results/README.md` · Nonlinear: `results/NONLINEAR_CHARGING_STUDY.md` · Learning: `results/REWARD_AND_LEARNING_STUDY.md`  
Figures: `results/figures/` (20 main) + 1 supplementary diagnostic · Catalog: `docs/FIGURE_CATALOG.md` · Tables: `results/tables/` · Raw: `results/raw/`

---

## Abstract

The electric vehicle routing problem with nonlinear charging (E-VRP-NL) couples customer sequencing with station choice and charge quantity under piecewise-linear, state-of-charge–dependent charging times. We study a hybrid architecture—HAT-EVRP-NL—that combines a static node encoder, a causal route-history Transformer, dynamic EV-state fusion, optional DeepSets fleet aggregation, and a hierarchical route/customer pointer, with **exact fixed-route nonlinear charging optimization** via `frvcpy` and independent Python physical validation. Policies are pretrained by behavior cloning and may be fine-tuned with REINFORCE; because `frvcpy` is non-differentiable, policy gradients use sampled routing-action log-probabilities and post-optimization rewards.

On the completed behavior-cloning then REINFORCE matrix (four architectures, seeds 42–46, one shared synthetic teacher set), official n=10 construction means after behavior cloning are 22.253 h (State Pointer), 22.379 h (Route Transformer), 22.060 h (Fleet DeepSets), and 22.065 h (RT + DeepSets). REINFORCE changes those means to 22.655, 22.287, 22.700, and 22.282 h. Only the Route Transformer mean improves, and only on 3 of 5 seeds. A nonlinear-versus-linear study on all 20 official instances at n=10, 20, and 40 shows that linear charging surrogates frequently become **infeasible under true Montoya physics** (e.g., 11/20 at n=40 fixed-route vs 20/20 TRUE-NL). Contribution: a rigorous experimental evaluation. These results do not establish global routing optimality or a reliable RL gain for every architecture.

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

**Figure 12.** Official n=10 construction (protocol P0; 20 instances × seeds 42–46) from the completed BC→RL matrix. Each point is one seed mean; the diamond is the mean of those five seeds. Behavior-cloning means: State Pointer 22.253 h, Route Transformer (RT) 22.379 h, Fleet DeepSets 22.060 h, RT + DeepSets 22.065 h. REINFORCE means: 22.655, 22.287, 22.700, and 22.282 h. Seed ranges overlap, so a lower mean is not a stable ranking.

![Figure 13. Paired effects](results/figures/fig13_history_deepsets_paired_effect.png)

**Figure 13.** Paired change on the same official n=10 construction instances: each point is one seed’s (RL mean − BC mean), and the diamond is the mean of the five seeds. Negative means the REINFORCE checkpoint has the lower full duration. Route Transformer is the only model whose mean change is negative (−0.092 h, better on 3/5 seeds). State Pointer, Fleet DeepSets, and RT + DeepSets are better on 1/5 seeds.

![Figure 14. Literature BKS gaps](results/figures/fig14_literature_bks_gap.png)

**Figure 14.** Mean gap of \(Z_\mathrm{lit}\) to Froger et al. (2021) Table 13 on official n=10 (20 instances, five seeds). Circles are behavior-cloning checkpoints and squares are REINFORCE checkpoints. Construction gaps are about 21–25%, one-shot local search about 14–17%, and hybrid insertion about 5–8%. The three panels use different search procedures, so they are separate measurements. These gaps do not support a state-of-the-art claim.

![Figure 15. Generalization](results/figures/fig15_generalization_scalability.png)

**Figure 15.** Construction-only transfer of the n=10 policy to official n=10, 20, 40, and 80 (20 instances at each size, five seeds). Solid lines are behavior cloning and dashed lines are REINFORCE. Panel B is (RT + DeepSets) − RT: the BC difference stays within about 0.6 h, while the RL difference at n=80 is +3.82 h (91.318 vs 87.501 h). Panel C is feasibility; every model and checkpoint is feasible on all evaluated instances, so the markers sit on 100%. The n=80 point is transfer of a policy trained at n=10.

![Figure 16. Search stages](results/figures/fig16_search_improvement_stages.png)

**Figure 16.** Official n=10 categorical comparison of construction, one-shot local search, and hybrid insertion (20 instances, mean of five seeds). The horizontal axis is limited to the observed range, about 19.8–22.7 h. The three protocols are alternatives with unequal search budgets, so a lower hybrid-insert mean is not an equal-budget ranking.

![Figure 17. Quality vs compute](results/figures/fig17_quality_vs_compute.png)

**Figure 17.** Official n=10 mean full duration against mean `frvcpy` oracle calls (20 instances, five seeds). Marker shape is the protocol and color is the architecture; a black edge marks the REINFORCE checkpoint. Construction and hybrid insertion cluster near 55 oracle calls, while one-shot local search is near 160. **Unequal protocol budgets, so horizontal position is not a controlled efficiency ranking.**

---

## 9. Results — Learning and Reward (RQ4)

![Figure 18. BC learning curves](results/figures/fig18_behavior_cloning_learning_curves.png)

**Figure 18.** Behavior-cloning validation full duration on the shared synthetic set (100 instances, evaluation every 25 steps through 1000 updates). Each panel is one architecture; faint lines are seeds 42–46 and the star is the best logged point of that seed. The curves drop early and then fluctuate.

![Figure 19. Controlled BC vs RL](results/figures/fig19_bc_vs_rl_checkpoint_quality.png)

**Figure 19.** Official n=10 construction means for the behavior-cloning checkpoint and the REINFORCE checkpoint (20/20 feasible). The vertical axis is reversed, so a lower duration is higher on the plot. One line is one seed. REINFORCE is lower on 1/5 State Pointer seeds, 3/5 Route Transformer seeds, 1/5 Fleet DeepSets seeds, and 1/5 RT + DeepSets seeds. This construction comparison does not establish an improvement under one-shot local search or hybrid insertion.

![Figure 20. Reward dynamics](results/figures/fig20_reinforce_reward_dynamics.png)

**Figure 20.** Normalized REINFORCE reward from the completed runs. Feasible returns are \(-Z_\mathrm{full}/(n T_\max)\). Faint lines are the five seeds and the bold line is their mean rolling median (window 5). The traces stay noisy through 300 updates. Read them together with the checkpoint outcomes in Figure 19 and with validation-based selection: validation preferred the REINFORCE checkpoint on 2/5, 3/5, 1/5, and 1/5 runs for State Pointer, Route Transformer, Fleet DeepSets, and RT + DeepSets.

Supplementary diagnostic: `results/supplementary/figures/fig21_rl_training_stability.png` (gradient norm, entropy, policy-gradient loss, and infeasible-rollout fraction; mean across five seeds).

Validation preferred the REINFORCE checkpoint on 7 of 20 runs. When it did not, the deployed checkpoint stayed at the behavior-cloning weights.

---

## 10. Ablations and Limitations

On official n=10 construction, architecture means differ by a few tenths of an hour and the seed ranges overlap. REINFORCE improves the Route Transformer mean and worsens the other three. Linear charging assumptions harm true-physics feasibility; `frvcpy` hybrid remains far more feasible than direct learned charging (4/20 vs 20/20 on tested n=10). Protocols P0, P1, and P2 use unequal search budgets. The n=80 curves transfer an n=10 policy onto 20 instances.

---

## 11. Conclusion

| RQ | Answer |
|---|---|
| RQ1 | On official n=10 construction, Route Transformer (22.379 h) does not beat Fleet DeepSets (22.060 h) after behavior cloning. |
| RQ2 | RT + DeepSets (22.065 h) is close to Fleet DeepSets at n=10 and is worse than Route Transformer at n=80 under the REINFORCE checkpoint (91.318 vs 87.501 h). |
| RQ3 | Yes—linear surrogates often fail true Montoya physics. |
| RQ4 | Official construction improves on 3/5 Route Transformer seeds and on 1/5 seeds for each other model. Validation keeps the REINFORCE checkpoint on 7/20 runs. |

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
