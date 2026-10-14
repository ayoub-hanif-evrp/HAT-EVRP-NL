# Learning Results Report

This report summarizes **what the training logs actually show** about neural learning,
without fabricating reward traces or claiming RL gains without evidence.

## Cycle-3 protocol (primary checkpoints)

- Independent synthetic generator (`v2`), n=10.
- Teacher: greedy/local-search routes with `frvcpy` charging.
- 400 behavior-cloning steps; 40 standard REINFORCE steps (`pomo_k=0`, no repair RL).
- 120 train / 12 validation synthetic instances; seeds 42–44.
- Checkpoint = best logged `(feasible_rate, −validation_objective)` — **not** the final RL iterate.

## Reward (implementation)

- Standard RL: `R = -Z_full` if feasible else `-100`; baseline = greedy rollout.
- Loss: `L = -(R - baseline) * sum(log π)`.
- `Z_full = travel + service + nonlinear charging`.
- Historical logs **do not** store sampled reward distributions.

## BC vs RL (synthetic validation)

| Model | Mean BC-best h | Mean RL-end h | Mean Δ | Seeds RL helped | Deployed from BC |
|---|---:|---:|---:|---:|---:|
| H1 | 15.5464 | 17.8534 | +2.3070 | 0/3 | 3/3 |
| H2 | 15.4235 | 22.0845 | +6.6610 | 1/3 | 2/3 |
| H3 | 15.6168 | 22.0632 | +6.4463 | 0/3 | 3/3 |
| H4 | 15.4325 | 26.4013 | +10.9688 | 0/3 | 3/3 |

### Key conclusions

1. Across seeds 42–44, RL **endpoint** beats best-BC validation on **1/3 H2 seeds** and **0/3 H4 seeds**. Mean RL endpoints are pulled up by collapses.
2. **H4 seed 42** is a clear instability (BC-best ≈15.71 h → RL-end ≈31.02 h). Checkpoint selection retains BC weights — published H4 is **not** an RL-improved model.
3. **Do not claim that RL improves HAT-EVRP-NL** from Cycle-3 alone. Even H2’s single-seed RL gain is not a robust multi-seed result.
4. Separate BC-only `.pt` files were not saved; official Montoya BC-only vs RL weight ablation remains **unavailable** without a controlled retrain.
5. Sampled reward traces are **unavailable** in historical logs (fig14).

## Figures

- Main paper learning figure: `results/figures/fig07_learning_and_reward.png`
- Supplementary historical plots: `results/supplementary/figures/fig11_*` … `fig17_*`

## Raw sources

- `results/raw/training_logs/training_H*_n10_seed*_c3.csv`
- `results/raw/training_logs/derived/`
- `results/raw/reward_learning/`
