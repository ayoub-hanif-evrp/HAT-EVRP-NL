# History-Aware Transformer Agents with DeepSets Coordination for Electric Vehicle Routing with Nonlinear Charging

## Abstract

We study Montoya’s electric vehicle routing problem with nonlinear charging
(E-VRP-NL) using a hybrid architecture: a shared causal visit-history
Transformer, optional DeepSets fleet aggregation, hierarchical customer
routing, and exact nonlinear charging via `frvcpy`. After correcting an
evaluation bug that made wall-clock classical restarts non-randomized, and
using equal construction and one-shot local-search protocols, we find that
history helps routing quality, but DeepSets does not consistently improve over
history alone. With fair one-shot local search, history-aware policies can
beat a strong classical greedy+LS baseline on official n=10–40 instances, yet
remain far from published Froger et al. (2021) best-known solutions
(~43% mean gap on n=10). A fully learned charging policy is largely
infeasible (4/20). The DeepSets coordination hypothesis is not supported.
**H2 (history only) is the strongest efficient neural variant; H4 remains the
full proposed architecture.**

## 1. Introduction

E-VRP-NL requires customer sequencing under battery and duration limits with
piecewise-linear concave charging curves (Montoya et al., 2017). Neural
construction heuristics have succeeded on classical VRP variants
(e.g., attention models), but nonlinear EV charging is usually handled by
specialized oracles. We ask whether sharing causal EV visit histories and
aggregating open routes with DeepSets (Zaheer et al., 2017) improves
coordination on Montoya’s static offline benchmark when charging is optimized
exactly.

## 2. Related work

Montoya et al. (2017) introduced E-VRP-NL and an ILS+HC matheuristic. Froger
et al. (2019, 2021) strengthened formulations and reported improved best-known
solutions (BKS) for the uncapacitated set. Fixed-route charging can be solved
by labeling (`frvcpy`). Deep Sets provide permutation-invariant set functions.
We do not claim novelty over these lines; we test one hybrid design under fair
computation.

## 3. Benchmark formulation

Instances: VRP-REP 2016-0020. Unlimited homogeneous EVs; full battery at
departure; serve each customer once; \(T_{\max}=10\) h; station-specific
piecewise-linear charging; partial recharge allowed; no time windows, cargo
limits, or station queues. Objective: total route duration (travel + service +
charging).

## 4. HAT-EVRP-NL architecture

Visit tokens encode node features, arrival/departure time and SOC, travel
energy, and charging amount/duration. A shared causal Transformer yields a
route-history vector \(h_t\) fused with explicit dynamic state \(s_t\). DeepSets
forms a fleet embedding \(g=\rho(\sum_i\phi(v_i))\) over open routes. A
hierarchical decoder selects an open EV or a new route, then a customer or
close action. `frvcpy` inserts nonlinear charging; an independent Python
forward simulator validates physics (SOC, duration, coverage). Ablations:
H1 state-only; H2 history; H3 DeepSets only; H4 history+DeepSets.

## 5. Training

Synthetic teachers only (official instances held out). Interleaved multi-open
imitation, optional REINFORCE, best validation checkpoint. Cycle-3 checkpoints
(`*_c3.pt`) are used throughout this final evaluation; no closure retraining.

## 6. Experimental protocol

**A.** Equal constructions (N samples, no LS).  
**B.** Fair one-shot LS: best-of-N then identical local search.  
**C.** Fair wall-clock: match measured neural time; classical uses genuinely
seeded RCL trials (`deterministic_first` only once).  

Seeds 42/43/44 on all 20 n=10 instances; seed 42 on all 20 n=20 and n=40; 3
n=80 instances (sample). BKS from Froger et al. (2021) Table 13. External Java
DIMACS checker not executed; **independent Python forward-simulation validation
only**. Paired comparisons use the 20 instances as units (seed-averaged), tie
tolerance \(10^{-9}\); wins+ties+losses = 100%.

## 7. Results

### 7.1 Equal constructions (n=10, seeds 42–44)

| Method | Mean h | Median h | Feasible | Runtime s | Oracle |
|---|---:|---:|---|---:|---:|
| H0 | 22.063 | 22.698 | 60/60 | 0.117 | 315 |
| H0_strong | **20.142** | 18.586 | 60/60 | 0.176 | 528 |
| H1 | 21.249 | 19.333 | 60/60 | 0.460 | 515 |
| H2 | 20.966 | 19.627 | 60/60 | 0.646 | 520 |
| H3 | 22.131 | 21.060 | 60/60 | 0.490 | 497 |
| H4 | 20.975 | 19.185 | 60/60 | 0.671 | 525 |

Without LS, H0_strong leads. Instance-paired: H2 vs H0_strong 20%/0%/80%;
H4−H2 40%/25%/35%.

### 7.2 Fair one-shot local search (primary; n=10)

| Method | Mean h | Median h | Routes | Charge stops | Charge h | Runtime s | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 22.063 | 22.698 | 2.867 | 4.300 | 1.863 | 0.107 | 315 |
| H0_strong | 20.142 | 18.586 | 2.800 | 3.650 | 1.535 | 0.156 | 528 |
| H1 | 19.993 | 18.429 | 2.950 | 3.433 | 1.485 | 0.555 | 750 |
| **H2** | **19.813** | **18.378** | 2.917 | 3.433 | 1.465 | 0.741 | 747 |
| H3 | 20.109 | 18.445 | 2.933 | 3.517 | 1.534 | 0.592 | 729 |
| H4 | 19.886 | 18.586 | 3.033 | 3.350 | 1.414 | 0.753 | 753 |

Instance-paired (`fair_one_shot_ls`, n=20 instances):

| Pair | Mean Δ (h) | 95% CI | Wins | Ties | Losses |
|---|---:|---|---:|---:|---:|
| H2−H1 | −0.180 | [−0.317, −0.044] | 80% | 10% | 10% |
| H3−H1 | +0.116 | [−0.151, +0.384] | 25% | 5% | 70% |
| **H4−H2** | **+0.074** | **[−0.018, +0.165]** | **20%** | **60%** | **20%** |
| H2−H0_strong | −0.329 | [−0.699, +0.041] | **65%** | **20%** | **15%** |
| H4−H0_strong | −0.255 | [−0.631, +0.120] | **50%** | **20%** | **30%** |

**H4 does not outperform H2.** History helps (H2≪H1). DeepSets alone does not
(H3). With fair LS, H2 improves on H0_strong on 65% of instances.

### 7.3 Scale (fair LS, seed 42)

| Size | H0_strong | H2 | H4 |
|---|---:|---:|---:|
| n=20 | 34.420 | 32.332 | 32.322 |
| n=40 | 60.195 | 57.173 | 57.506 |

### 7.4 Corrected wall-clock (seed 42)

| Method | Mean h | Runtime s | Oracle |
|---|---:|---:|---:|
| H0_strong | 19.895 | 0.412 | 2466 |
| **H2** | **19.586** | 0.448 | 760 |
| H4 | 19.840 | 0.458 | 745 |

Unique classical constructions ≈40+ per instance under the fixed RNG protocol.
Superseded non-randomized wall-clock runs are excluded from conclusions.

### 7.5 BKS gaps and learned charging

n=10 fair LS vs Froger et al. (2021) Table 13: H0_strong 44.8%, H2 42.7%, H4
43.2% mean gap. Direct CHARGE: 4/20 feasible. Hybrid routing + `frvcpy`: 20/20.

## 8. Limitations

CPU-only; incomplete 160/320 evaluation; no external Java checker; large BKS
gaps; neural methods use more oracle/runtime than H0_strong under fair LS;
results are far from specialized metaheuristics.

## 9. Conclusion

Transformer visit history improves neural routing. DeepSets fleet coordination
does not meaningfully improve Montoya E-VRP-NL under our fair tests—**H2 is the
strongest efficient experimental variant; H4 is the complete proposed
architecture without a consistent win over H2**. Exact nonlinear charging
remains necessary. Authoritative numbers: `results/FINAL_REPORT.md`.

## References

- Montoya, J.A., Guéret, C., Mendoza, J.E., Villegas, J.G. (2017). The electric
  vehicle routing problem with nonlinear charging function. *Transportation
  Research Part B*, 103, 87–110.
- Froger, A., Jabali, O., Mendoza, J.E., Laporte, G. (2021). The electric
  vehicle routing problem with capacitated charging stations. Working paper /
  HAL; BKS Table 13 used for uncapacitated E-VRP-NL.
- Froger, A., Mendoza, J.E., Jabali, O., Laporte, G. (2019). Improved
  formulations for E-VRP-NL. *Computers & Operations Research*.
- Zaheer, M., et al. (2017). Deep Sets. *NeurIPS*.
- Kullman, N., et al. `frvcpy`: fixed-route vehicle charging problem solver.
