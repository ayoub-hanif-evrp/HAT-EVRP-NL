# Reward definitions used in train.py

- Objective convention: Training reward (not the publication BKS gap)
- Units: hours / dimensionless penalty
- Benchmark / data: any decode size
- Instance / seed count: n/a
- Methods / checkpoints: standard RL / POMO / repair RL
- Protocol: as implemented in train.py
- Seeds: n/a
- Source raw: `train.py (reinforce / pomo_reinforce_step / repair_reinforce_step)`

| Stage | Feasible reward | Infeasible penalty | Baseline | Loss | Used for Cycle-3 H2/H4? |
|---|---|---|---|---|---|
| Behavior cloning | N/A (cross-entropy on teacher actions) | N/A | N/A | masked hierarchical CE | Yes (400 steps) |
| Standard REINFORCE | R = −Z_full | R = −100 | greedy rollout reward | L = −(R−b)·Σ log π | Yes (40 steps) |
| POMO REINFORCE | R = −Z_full | R = −100 | mean of K rollout rewards | mean of −(R−b)·log π | No (Cycle-3 pomo_k=0) |
| Repair REINFORCE | R = Z_before − Z_after | R = −50 | mean repair-rollout reward | mean of −(R−b)·log π | No (Cycle-3 repair_rl_steps=0) |

## Interpretation

Z_full = travel + service + nonlinear charging. Service is constant for complete feasible coverage, so maximizing R over feasible solutions is equivalent to minimizing the literature travel+charging objective. Fixed penalties −100/−50 were not re-tuned by size; all-infeasible repair batches yield zero gradient when advantages vanish.
