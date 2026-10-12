# History-Aware Transformer Agents with DeepSets Coordination for Electric Vehicle Routing with Nonlinear Charging

## Abstract

We study whether a shared causal visit-history Transformer combined with a
permutation-invariant DeepSets fleet representation can improve customer
allocation and sequencing for the Montoya electric vehicle routing problem with
nonlinear charging (E-VRP-NL). Charging decisions are handled by an exact
fixed-route nonlinear charging optimizer (`frvcpy`), while neural policies
construct customer sequences. Across three training seeds on all twenty
10-customer official instances, history-aware policies outperform state-only
ablations. DeepSets is correctly implemented and receives meaningful multi-open
training signal, but yields no reliable objective gain over the history-only
model. Under equal construction samples, a strong classical greedy+local-search
baseline remains best. Under equal local-search refinement applied to neural
routes, history-aware neural construction beats the strong classical baseline
(H4 mean 19.597 h vs 20.142 h; H4 better on 76.7% of paired comparisons). A
fully learned hierarchical charging policy remains largely infeasible (4/20)
and is reported as a negative ablation. We conclude that history-aware
construction is useful for E-VRP-NL, while DeepSets coordination and learned
nonlinear charging are not yet beneficial in this static offline setting.

## 1. Introduction and motivation

Electric vehicle routing with nonlinear charging requires jointly deciding
customer sequences and physically valid charging plans. Classical methods
combine construction heuristics with specialized charging optimization.
Learning-based routing can propose customer orders, but must respect battery,
duration, and nonlinear charging physics.

Our research question is empirical and architectural: does sharing causal
visit-history representations across routes, aggregated by DeepSets into a
global fleet state, improve coordinated customer assignment for Montoya
E-VRP-NL when charging is optimized exactly?

## 2. Montoya E-VRP-NL formulation

We use the VRP-REP 2016-0020 instances of Montoya et al. (2017):

- one depot; unlimited homogeneous EVs; full battery at departure;
- serve each customer exactly once; service times included;
- maximum route duration \(T_{\max}=10\) h;
- piecewise-linear concave station-specific charging curves;
- partial charging and repeated station visits allowed;
- no customer time windows, cargo capacities, or station queues;
- objective: minimize total route duration (travel + service + charging).

## 3. HAT-EVRP-NL architecture

The policy is hierarchical and hybrid:

1. **Shared causal Transformer** encodes each EV’s visit history (node
   embedding, arrival/departure time and SOC, travel energy, charging
   amount/duration, node type).
2. **Dynamic state fusion** combines the last Transformer hidden vector with
   explicit current-route scalars.
3. **DeepSets** forms \(g=\rho(\sum_i\phi(v_i))\) over open/closed EV states.
4. **Hierarchical decoder** selects an open EV or a new route, then the next
   customer or route closure.
5. **Nonlinear charging oracle** (`frvcpy`) inserts stations and charge
   quantities for each completed customer prefix.
6. **Independent checker** validates SOC, charging durations, coverage, and
   \(T_{\max}\).

Ablations: H1 state-only; H2 history without DeepSets; H3 DeepSets without
history; H4 full model.

## 4. Training and charging methodology

Synthetic instances matching Montoya characteristics generate verified teacher
solutions via randomized greedy + charging optimization. Official instances are
evaluation-only. Training uses interleaved multi-open-route imitation so several
routes compete for remaining customers, optionally followed by REINFORCE
fine-tuning with best-validation checkpoint selection.

A secondary hierarchical GO/CHARGE/station/target-SOC policy is trained from
oracle charging labels using exact curves and physical Wh candidates. It is
evaluated separately from the hybrid method and never credited for
`frvcpy`-repaired solutions.

## 5. Experimental settings

- Hardware: CPU PyTorch; seeds 42/43/44; versioned `c3` checkpoints.
- Final training: 120 synthetic teachers, 400 BC + 40 RL updates.
- Evaluation: all 20 n=10 instances; all 20 n=20; all 20 n=40 (equal LS);
  3 n=80 instances (limited CPU sample).
- Budgets: equal construction samples; equal local-search passes/budget;
  wall-clock equalization (seed 42).
- Baselines: H0 randomized greedy; H0_strong greedy + 2-opt/relocate + oracle.

Verified published BKS values were not available in-repo; gaps are `NA`.

## 6. Results

### 6.1 Ten-customer official set (pooled 60 evaluations)

Equal samples: H0 22.063; H0_strong 20.142; H1 21.249; H2 20.966; H3 22.131;
H4 20.975 (all 60/60 feasible). H4−H2 mean delta +0.009 h.

Equal LS: H1 19.644; H2 19.601; H3 19.702; H4 19.597 vs H0_strong 20.142.
H4 beats H0_strong on 76.7% of pairs.

Wall-clock (seed 42): H0_strong 20.010; H2 20.792; H4 21.057 — classical wins
when time, not sample count, is equalized.

### 6.2 Scalability

With equal LS, zero-shot n=20 means: H0_strong 34.420; H2 32.217; H4 32.307
(20/20). n=40: 60.195 / 57.173 / 57.506 (20/20). n=80 sample (3 instances):
95.506 / 88.151 / 88.151.

### 6.3 DeepSets ablation

Diagnostics confirm permutation invariance and nonzero DeepSets gradients.
Paired H4 vs H2 differences remain near zero. History is the dominant neural
ingredient; fleet aggregation is not shown to improve this static problem.

### 6.4 Learned charging

Direct CHARGE feasibility stays 4/20. Hybrid neural routing + `frvcpy` is
20/20. Repair results equal hybrid objectives and must not be attributed to
learned charging.

## 7. Limitations

CPU-only experiments; incomplete large-instance coverage; unavailable verified
BKS table; wall-clock comparisons sensitive to implementation speed; RL
fine-tuning often fails to improve validation objectives.

## 8. Conclusion

A history-aware Transformer hybridized with exact nonlinear charging
optimization produces feasible E-VRP-NL solutions and, when paired with the
same local-search budget as classical methods, improves upon a strong greedy
baseline on n=10–40 instances in our experiments. DeepSets global coordination
and fully learned nonlinear charging do not currently provide reliable gains.
Future work should target larger-scale training, stronger classical teachers,
and charging policies that remain feasible under route-duration constraints
without oracle repair.
