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
route-history vector fused with explicit state. DeepSets forms
\(g=\rho(\sum_i\phi(v_i))\). A hierarchical decoder selects an open EV or a new
route, then a customer or close action. `frvcpy` inserts charging; an
independent checker validates physics. Ablations: H1 state-only; H2 history;
H3 DeepSets only; H4 history+DeepSets.

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
DIMACS checker not executed; Python validation only.

## 7. Results

Equal constructions (n=10): H0_strong 20.142; H2 20.966; H4 20.975.  
Fair LS: H2 19.813; H4 19.886; H0_strong 20.142 (oracle ~747 vs 528).  
Instance-paired H4−H2: mean +0.074 h; H4 better 30%, tie 60%, H2 better 20%.  
Wall-clock (seed 42): H2 19.586; H4 19.840; H0_strong 19.895; ~40+ unique
classical constructions per instance.  
n=20/40 fair LS: H2/H4 beat H0_strong; H4 does not dominate H2.  
n=10 BKS gaps ≈43% (H2/H4). Direct CHARGE: 4/20 feasible.

## 8. Limitations

CPU-only; incomplete 160/320 evaluation; no external Java checker; large BKS
gaps; neural methods use more oracle/runtime than H0_strong under fair LS.

## 9. Conclusion

History-aware hybrid routing is useful. DeepSets fleet coordination does not
meaningfully improve Montoya E-VRP-NL under our fair tests. Exact nonlinear
charging remains necessary. Authoritative numbers: `results/FINAL_REPORT.md`.

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
