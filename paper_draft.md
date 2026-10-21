# History-Aware Neural Electric Vehicle Routing with DeepSets Fleet Coordination and Exact Nonlinear Charging Optimization

**HAT-EVRP-NL — research manuscript**  
Results: `results/README.md` · Nonlinear: `results/NONLINEAR_CHARGING_STUDY.md` · Learning: `results/REWARD_AND_LEARNING_STUDY.md`  
Figures: `results/figures/` (20 main) + 1 supplementary diagnostic · Catalog: `docs/FIGURE_CATALOG.md` · Tables: `results/tables/` · Raw: `results/raw/`

---

## Abstract

The primary method is a neural constructor trained in two phases: behavior cloning, then REINFORCE. Charging is not learned. After the policy chooses a customer sequence, `frvcpy` returns the exact nonlinear charging plan, and an independent checker tests feasibility. Local search, ruin-and-repair, and hybrid insertion are not part of this training loop and are not part of the primary comparison. Those searches remain supplementary.

The question the primary numbers answer is whether REINFORCE improves the behavior-cloning constructor. On the completed five-seed matrix it does not, in the sense required here. Official n=10 construction means after behavior cloning are 22.253 h (State Pointer), 22.379 h (Route Transformer), 22.060 h (Fleet DeepSets), and 22.065 h (RT + DeepSets). REINFORCE changes those means to 22.655, 22.287, 22.700, and 22.282 h. The Route Transformer paired change is −0.091 h, interval [−0.336, +0.154], which includes zero, and that checkpoint is not better under hybrid insertion. Every recorded RL rollout in that matrix is feasible, so the phase is not learning feasibility. Entropy falls during REINFORCE, and validation duration at the last update is worse than the behavior-cloning checkpoint.

A sampled log-ratio penalty has expected gradient zero and is not a test of anchoring. The corrected penalty is \(0.01\,\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{BC}})\) on the masked route and customer distributions. Rerun from the same checkpoints, Route Transformer validation duration rises by 0.045 h, interval [−0.109, +0.199], better on 2/5 seeds. RT + DeepSets rises by 0.049 h, interval [−0.076, +0.174], better on 2/5 seeds. Official evaluation of that variant was not started. A separate charging study finds that linear charging surrogates frequently become infeasible under true Montoya physics. These results do not establish a reliable RL gain.

---

## 1. Introduction

Battery-electric fleets introduce charging time that is *nonlinear* in remaining capacity. Montoya et al. (2017) formalized E-VRP-NL with station-specific piecewise-linear charging curves and a maximum route duration.

The Montoya E-VRP-NL benchmark is particularly appropriate for evaluating the proposed hybrid model because it explicitly represents charging-time nonlinearity through station-specific piecewise-linear curves. Consequently, the charging duration required for a given energy increment depends on the battery state of charge. We therefore examine not only total routing cost, but also the physical feasibility, charging time, and routing decisions obtained when linear charging approximations are used.

**RQ1.** Does the Route Transformer improve official n=10 construction relative to the State Pointer?  
**RQ2.** Does adding Fleet DeepSets improve upon the Route Transformer alone?  
**RQ3.** Does nonlinear charging optimization matter versus linear charging assumptions?  
**RQ4.** Does REINFORCE improve the behavior-cloning checkpoint on the predeclared comparison: Route Transformer, official n=10, construction only?

The other architecture, size, and protocol contrasts in the 48-cell table are exploratory. Five seeds that share one synthetic training set measure initialization and sampling noise, not a new draw of the training data.

---

## 2. Related Work

Montoya et al. (2017) formulate the electric vehicle routing problem with nonlinear charging: routes may detour to stations whose charging time is a piecewise-linear function of state of charge, and each route is limited by a maximum duration. Froger et al. (2021) give the best known solutions used here, compared through the literature objective that removes service time. Kullman et al. (2021) provide `frvcpy`, the fixed-route labeling solver this project calls after a customer sequence has been chosen. The neural policy does not replace that solver.

On the learning side, Kool et al. (2019) train an attention-based construction policy with REINFORCE, and Kwon et al. (2020) reduce gradient variance with multiple rollouts from one instance. Those models choose the next node in a complete graph. The policy here is narrower: it chooses which route to extend and which customer to append, then asks `frvcpy` for the charging plan. Chung, Lee, and Tsang (2025) survey this family of neural combinatorial methods. The present experiments ask whether a route-context encoder, a fleet aggregator, and a short REINFORCE stage change official durations. They do not claim a new state of the art.

---

## 3. Problem Formulation

An instance has one depot \(0\), customers \(C\), and charging stations \(S\). Travel time and energy between nodes are fixed. Each customer \(i\) has a service time. Every vehicle starts and ends at the depot, carries a battery of capacity \(Q\), and must finish within a route limit \(T_{\max}\). Each customer is served by exactly one visit. Stations may be used more than once, including more than once on the same route, and the amount charged may be any feasible partial recharge.

At a station of type \(s\), the time to charge from energy \(q_{\mathrm{in}}\) to \(q_{\mathrm{out}}\) is the difference of the station's piecewise-linear cumulative curve, \(F_s(q_{\mathrm{out}})-F_s(q_{\mathrm{in}})\). State of charge after a charge cannot exceed \(Q\), and it must stay high enough to reach the next station or the depot. A solution is feasible only if every route satisfies these physical checks. The objective used for learning and for the tables is

\[
Z_\mathrm{full}=T_\mathrm{travel}+T_\mathrm{service}+T_\mathrm{charge}.
\]

The literature gap uses \(Z_\mathrm{lit}=Z_\mathrm{full}-T_\mathrm{service}\), which matches Froger et al. when every customer is served. `frvcpy` chooses the stations and charge amounts for a fixed customer sequence. The learned policy chooses only the customer sequence.

A request is one call to that solver for a customer sequence. A cache hit returns a sequence that was already solved. An execution is a cache miss, so it is one new labeling solve. Official construction uses about \(n(n+1)/2\) executions (55, 210, 819, and 3,228 at n=10, 20, 40, and 80), which is the cost of testing candidate appends. Local search and ruin-and-repair add further executions on new sequences. The routing choices themselves remain combinatorial. The labeling solve is the expensive inner step, and it is not differentiated.

---

## 4. Proposed HAT-EVRP-NL Method

![Figure 1. HAT-EVRP-NL architecture](results/figures/fig01_hat_evrp_nl_architecture.png)

**Figure 1.** Hybrid architecture. Blue blocks are learned: a node encoder, the Route Transformer (RT) over optimizer-enriched stops, dynamic state fusion, optional Fleet DeepSets, and a hierarchical pointer. Orange blocks are not differentiated: `frvcpy` builds the charging plan for the current customer sequence, and a Python checker tests physical feasibility. A causal attention mask stops a stop token from attending to later stops in that plan. The plan itself can contain stations and charge amounts chosen by the solver, so the encoded context is not a log of committed neural actions. Training is behavior cloning, then REINFORCE. Gradients pass through the routing log-probabilities only.

Ablations (display names): State Pointer · Route Transformer (RT) · Fleet DeepSets · Route Transformer + DeepSets (HAT-EVRP-NL). Classical: Greedy Construction / Greedy + Local Search. Internal checkpoint keys H0–H4 are retained for reproducibility only (`presentation_names.py`).

![Figure 2. Montoya layouts](results/figures/fig02_montoya_instance_layouts.png)

**Figure 2.** Official Montoya spatial layouts for location families tc0, tc1, and tc2 (n=20 representatives). Markers: depot (square), customers (circles), charging stations by technology (triangles). Coordinates are dataset units from the XML files.

![Figure 3. Feasible EV routes](results/figures/fig03_feasible_ev_routes.png)

**Figure 3.** Validated hybrid solution for `tc2c40s5cf3` (Route Transformer (RT), hybrid_insert; full duration 40.14 h). Routes from saved JSON; open circles mark charge visits. Source: `results/raw/regression_recovery/solutions/`.

---

## 5. Training Methodology

The completed study uses two training phases and no others. Behavior cloning imitates feasible customer sequences from randomized greedy construction and local search. For a teacher trajectory \(\tau\), the recorded loss is the mean next-action cross-entropy

\[
L_{\mathrm{BC}}=-\frac{1}{|\tau|}\sum_{t\in\tau}\log \pi_\theta(a_t\mid s_t).
\]

The action \(a_t\) is hierarchical. The policy first picks an open route or the decision to open a new route, then picks a customer or the decision to close the route:

\[
\pi_\theta(a\mid s)=\pi_\theta(r\mid s)\,\pi_\theta(c\mid r,s).
\]

REINFORCE then samples \(K=4\) rollouts on a training instance. The normalized reward is \(R=-Z_\mathrm{full}/(n T_{\max})\) when the rollout is feasible. An infeasible rollout scores \(R=-1-0.05-\) (unserved fraction), plus a violation penalty only when that quantity is already finite. The baseline of rollout \(k\) is the mean reward of the other rollouts. With advantage \(A_k=R_k-b_k\),

\[
L_{\mathrm{RL}}=-\frac{1}{K}\sum_{k=1}^{K} A_k \log \pi_\theta(\tau_k).
\]

The entropy coefficient in this matrix is zero, so entropy is logged and is not added to the loss. The older Cycle-3 reward, \(R=-Z_\mathrm{full}\) or \(R=-100\), was not used here. `frvcpy` stays outside the gradient.

The original 20 runs drew each minibatch with the cyclic index \((t-1+b)\bmod 500\). At 300 RL updates and batch size 2 that formula touches indices 0 through 300 only, so 199 of the 500 training instances never enter RL. Behavior cloning, with 1,000 updates, does cycle through the full set. A corrected sampler shuffles the 500 instances and finishes a pass before repeating. The corrected RL runs start from the saved behavior-cloning checkpoints, use the same 300 updates, reward, and leave-one-out baseline, and write to a separate directory. They do not replace the original runs.

On the frozen comparison, shuffled RL does not lower synthetic validation duration relative to the original cyclic RL. For Route Transformer the paired difference is +0.035 h, interval [−0.046, +0.116], with the shuffled checkpoint lower on 1/5 seeds. For RT + DeepSets it is +0.044 h, interval [−0.033, +0.121], lower on 2/5 seeds. Both intervals include zero. Shuffled RL is better than its own behavior-cloning checkpoint on 1/5 Route Transformer seeds and on 0/5 RT + DeepSets seeds. Covering the unused training instances does not establish an RL improvement. The same logs show a second, more specific failure: entropy falls on every seed, and validation duration at update 300 is worse than behavior cloning on all 20 original runs. A follow-up that added \(0.01(\log\pi_\theta(a)-\log\pi_{\mathrm{BC}}(a))\) for the sampled action is not a KL test: the expected gradient of that term is zero. Its validation changes, +0.068 h and +0.125 h, stay in `results/raw/bc_rl_kl_v1` and are not used as evidence about anchoring. The corrected penalty is \(0.01\,\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{BC}})\) on the masked route and customer distributions, written to `results/raw/bc_rl_kl_v2`. Route Transformer validation duration rises by 0.045 h, interval [−0.109, +0.199], better on 2/5 seeds. RT + DeepSets rises by 0.049 h, interval [−0.076, +0.174], better on 2/5 seeds. The predeclared rule for official neural-construction evaluation was not met, so that evaluation was not run. The diagnosis is in `docs/RL_DIAGNOSIS.md`.

Details: `docs/TRAINING_AND_REWARD_METHODOLOGY.md`.

---

## 6. Experimental Setup

The five-seed BC→RL study and the earlier Cycle-3 comparison are separate experiments.

The BC→RL matrix trains State Pointer, Route Transformer, Fleet DeepSets, and RT + DeepSets on one shared synthetic n=10 set: 500 training instances, 100 validation instances, data seed 1000, model seeds 42–46. Each run uses 1,000 behavior-cloning updates, then 300 REINFORCE updates with 4 leave-one-out rollouts. Official Montoya evaluation scores `bc_best` and `rl_best` separately at n=10, 20, 40, and 80, with 20 instances at each size. P0 is construction only. P1 is one-shot basic local search (1 pass, budget 800). P2 is hybrid insertion: classical ruin for 6 iterations with budget 600, then neural insertion for 6 iterations. Checkpoints are selected on the synthetic validation set. The official instances are not used to choose hyperparameters or checkpoints.

Cycle-3 fair local search used seeds 42–44 and the legacy reward. The nonlinear-charging study covers all 20 official instances at n=10, 20, and 40. Its recovered n=40 hybrid figure is a single-seed recovery, not a cell of the five-seed matrix. Metrics for the new study are feasibility, \(Z_\mathrm{full}\), \(Z_\mathrm{lit}\), frvcpy executions, requests, cache hits, and wall-clock time. Hybrid-insertion executions and runtime include construction, classical ruin, and neural repair.

The equal-budget classical comparison uses the same official instances and does not retrain a network. For each size, the execution budget is the mean number of solver executions recorded by the neural protocols: about 55, 162, and 418 at n=10 for construction, one-shot local search, and hybrid insertion. Deterministic cheapest insertion runs first on a shared oracle and stops before a solve that would pass the cap. Greedy plus local search then restarts from a randomized candidate list and applies basic local search until the cap. Classical ruin-and-repair uses a new seed on each chunk until the same cap. Realized executions, requests, cache hits, and elapsed time are stored separately. A feasible-only mean is not compared with a fully feasible neural mean unless both sides cover the same instances. Wall-clock budgets use the neural mean runtime of the same protocol and are a separate arm from the execution cap.

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

**Figure 17.** Official n=10 mean full duration against mean `frvcpy` solver executions (20 instances, five seeds). Marker shape is the protocol and color is the architecture; a black edge marks the REINFORCE checkpoint. Construction is about 55 executions, one-shot local search about 162, and hybrid insertion about 418. The hybrid count includes construction, classical ruin, and neural repair. **Unequal protocol budgets, so horizontal position is not a controlled efficiency ranking.**

The matched-budget comparison gives each classical method the mean execution count of that neural protocol. Full rows are in `results/tables/classical_matched_v1_summary.csv`. Realized execution counts match these budgets at every size. Requests and cache hits are larger than executions, especially for ruin-and-repair, and are not the budget.

| Size | Budget (executions) | Greedy + LS feasible | Greedy + LS hours | Ruin-and-repair hours | RT behavior cloning |
|---|---|---|---|---|---|
| 10 | 55 | 12/20 | 20.153 on those 12 | same construction | 22.379 on all 20; 20.135 on the same 12 |
| 10 | 162 | 20/20 | 21.551 | 22.938 | 20.997 |
| 10 | 418 | 20/20 | 21.068 | 21.433 | 19.974 |
| 20 | 210 | 17/20 | 35.430 on those 17 | same construction | 33.521 on all 20; 32.690 on the same 17 |
| 20 | 660 | 20/20 | 35.358 | 36.655 | 32.628 |
| 20 | 890 | 20/20 | 35.318 | 36.636 | 31.828 |
| 40 | 819 | 0/20 | no feasible mean | same construction | 60.897 on all 20 |
| 40 | 1,539 | 20/20 | 62.862 | 62.482 | 59.744 |
| 40 | 1,932 | 20/20 | 62.862 | 62.482 | 59.663 |
| 80 | 3,228 | 0/20 | no feasible mean | same construction | 88.944 on all 20 |
| 80 | 3,998 | 20/20 | 92.842 | 92.941 | 88.365 |
| 80 | 5,200 | 20/20 | 92.842 | 92.941 | 88.176 |

At the construction budget, deterministic greedy often leaves customers unserved: 12/20 at n=10, 17/20 at n=20, 0/20 at n=40 (819 executions), and 0/20 at n=80 (3,228 executions). On the n=10 instances both methods finish, the durations are close (20.153 h versus 20.135 h). Once the budget is large enough for classical search to finish every instance, Route Transformer behavior cloning remains lower: about 0.55 h at 162 executions and about 1.09 h at 418 executions for n=10, about 2.7 h and 3.5 h at the n=20 search budgets, about 3.1 h at both n=40 search budgets, and about 4.5 h at both n=80 search budgets. Extra executions beyond the one-shot budget do not change the classical means at n=40 or n=80. Classical ruin-and-repair does not beat greedy plus local search at these caps. This is an execution-matched comparison. It does not by itself make the method a state of the art solver.

The wall-clock arm gives classical search the mean neural runtime of the same protocol. At n=10 that time allows far more than 55–418 solver executions, because the neural runtime includes the policy. Greedy plus local search is feasible on 19/20 instances at the 0.045 s construction budget (20.902 h on those 19) and on 20/20 at the longer n=10 budgets. Ruin-and-repair finishes all 20. Its means are 21.749 h, 20.054 h, and 19.527 h at the construction, one-shot, and hybrid time budgets, against Route Transformer behavior-cloning means of 22.379 h, 20.997 h, and 19.974 h. At n=20, 40, and 80 the same time budgets still leave Route Transformer lower: about 33.5 h versus 36.2 h at n=20 construction, about 60.9 h versus 62.5 h at n=40, and about 88.9 h versus 92.8 h at n=80. Rows for both arms are in `results/tables/classical_matched_v1_summary.csv`.

---

## 9. Results — Learning and Reward (RQ4)

![Figure 18. BC learning curves](results/figures/fig18_behavior_cloning_learning_curves.png)

**Figure 18.** Behavior-cloning validation full duration on the shared synthetic set (100 instances, evaluation every 25 steps through 1000 updates). Each panel is one architecture; faint lines are seeds 42–46 and the star is the best logged point of that seed. The curves drop early and then fluctuate.

![Figure 19. Controlled BC vs RL](results/figures/fig19_bc_vs_rl_checkpoint_quality.png)

**Figure 19.** Official n=10 construction means for the behavior-cloning checkpoint and the REINFORCE checkpoint (20/20 feasible). The vertical axis is reversed, so a lower duration is higher on the plot. One line is one seed. REINFORCE is lower on 1/5 State Pointer seeds, 3/5 Route Transformer seeds, 1/5 Fleet DeepSets seeds, and 1/5 RT + DeepSets seeds. This construction comparison does not establish an improvement under one-shot local search or hybrid insertion.

Paired differences for every model, size, and protocol are in `results/tables/bc_rl_full_v1_paired.csv`. Each seed contributes its mean over 20 instances. The interval is a Student \(t\) 95% interval on those five differences (RL − BC, in hours). Negative means the REINFORCE checkpoint has the lower duration. All 9,600 official rows are feasible.

| Size | Protocol | State Pointer | RT | Fleet DeepSets | RT + DeepSets |
|---|---|---|---|---|---|
| 10 | P0 | +0.402 [−0.413, +1.216] (1/5) | −0.091 [−0.336, +0.154] (3/5) | +0.641 [−0.540, +1.822] (1/5) | +0.217 [−0.241, +0.676] (1/5) |
| 10 | P1 | +0.303 [−0.204, +0.810] (1/5) | −0.059 [−0.249, +0.132] (3/5) | +0.618 [−0.282, +1.518] (1/5) | +0.301 [−0.069, +0.671] (1/5) |
| 10 | P2 | −0.096 [−0.243, +0.051] (4/5) | +0.038 [−0.019, +0.095] (1/5) | +0.192 [+0.022, +0.362] (0/5) | +0.075 [−0.129, +0.279] (2/5) |
| 20 | P0 | +0.183 [−0.290, +0.656] (1/5) | +0.065 [−0.563, +0.693] (2/5) | −0.088 [−0.524, +0.347] (3/5) | +0.385 [−0.363, +1.132] (1/5) |
| 20 | P1 | +0.244 [−0.254, +0.743] (1/5) | +0.186 [−0.374, +0.745] (2/5) | +0.100 [−0.218, +0.417] (2/5) | +0.387 [−0.237, +1.011] (1/5) |
| 20 | P2 | +0.394 [−0.451, +1.238] (1/5) | +0.125 [−0.484, +0.734] (2/5) | +0.143 [−0.556, +0.843] (3/5) | +0.671 [−0.172, +1.514] (1/5) |
| 40 | P0 | −0.272 [−1.432, +0.888] (3/5) | −1.076 [−3.495, +1.342] (2/5) | +0.296 [−1.353, +1.944] (2/5) | −0.039 [−0.760, +0.681] (2/5) |
| 40 | P1 | −0.339 [−1.609, +0.932] (2/5) | −1.169 [−3.700, +1.361] (2/5) | +0.033 [−1.549, +1.615] (2/5) | −0.031 [−0.755, +0.694] (2/5) |
| 40 | P2 | −0.193 [−1.310, +0.923] (3/5) | −1.231 [−3.909, +1.446] (3/5) | +0.458 [−0.901, +1.817] (2/5) | +0.049 [−0.277, +0.375] (2/5) |
| 80 | P0 | +0.463 [−0.332, +1.259] (2/5) | −1.443 [−4.506, +1.619] (2/5) | −0.171 [−1.559, +1.218] (2/5) | +3.007 [−3.510, +9.524] (1/5) |
| 80 | P1 | +0.385 [−0.449, +1.219] (3/5) | −1.505 [−4.640, +1.630] (2/5) | −0.189 [−1.623, +1.245] (2/5) | +2.986 [−3.571, +9.543] (1/5) |
| 80 | P2 | +0.264 [−0.438, +0.967] (2/5) | −1.417 [−4.388, +1.554] (2/5) | −0.258 [−1.243, +0.727] (3/5) | +1.943 [−2.760, +6.646] (2/5) |

Forty-seven of the 48 intervals include zero. The exception is Fleet DeepSets under n=10 hybrid insertion, where REINFORCE is worse on all five seeds. The Route Transformer construction dip at n=10 does not carry over to hybrid insertion. Wider negative point estimates at n=40 and n=80 still have intervals that include zero.

![Figure 20. Reward dynamics](results/figures/fig20_reinforce_reward_dynamics.png)

**Figure 20.** Normalized REINFORCE reward from the completed runs. Feasible returns are \(-Z_\mathrm{full}/(n T_\max)\). Faint lines are the five seeds and the bold line is their mean rolling median (window 5). The traces stay noisy through 300 updates. Read them together with the checkpoint outcomes in Figure 19 and with validation-based selection: validation preferred the REINFORCE checkpoint on 2/5, 3/5, 1/5, and 1/5 runs for State Pointer, Route Transformer, Fleet DeepSets, and RT + DeepSets.

Supplementary diagnostic: `results/supplementary/figures/fig21_rl_training_stability.png` (gradient norm, entropy, policy-gradient loss, and infeasible-rollout fraction; mean across five seeds).

Validation preferred the REINFORCE checkpoint on 7 of 20 runs. When it did not, the deployed checkpoint stayed at the behavior-cloning weights.

---

## 10. Ablations and Limitations

On official n=10 construction, architecture means differ by a few tenths of an hour and the seed ranges overlap. A lower Route Transformer mean after REINFORCE is a five-seed point estimate whose paired interval includes zero, about 0.4% of the behavior-cloning duration, and it reverses under hybrid insertion. Linear charging assumptions harm true-physics feasibility; `frvcpy` hybrid remains far more feasible than direct learned charging (4/20 vs 20/20 on the earlier n=10 charging comparison). Protocols P0, P1, and P2 use unequal search budgets; the table above is the execution-matched reading. The n=80 curves transfer an n=10 policy onto 20 instances. All five seeds share one synthetic training set, so the intervals do not measure a change of training data. P0, P1, and P2 on one size reuse the same instances and checkpoints.

---

## 11. Conclusion

| RQ | Answer |
|---|---|
| RQ1 | On official n=10 construction, Route Transformer (22.379 h) does not beat Fleet DeepSets (22.060 h) after behavior cloning. |
| RQ2 | RT + DeepSets (22.065 h) is close to Fleet DeepSets at n=10 and is worse than Route Transformer at n=80 under the REINFORCE checkpoint (91.318 vs 87.501 h). |
| RQ3 | Yes—linear surrogates often fail true Montoya physics. |
| RQ4 | Not established. The predeclared Route Transformer n=10 construction change is −0.091 h, interval [−0.336, +0.154], and hybrid insertion reverses the sign. A corrected 0.01 distribution-KL penalty also fails its validation rule (+0.045 h and +0.049 h, each better on 2/5 seeds). The other 46 cells are exploratory; one interval excluding zero is not a confirmed discovery under 48 comparisons. |

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
