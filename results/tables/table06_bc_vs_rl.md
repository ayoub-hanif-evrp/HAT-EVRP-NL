# Behavior cloning vs reinforcement learning

- Objective: Full duration (h)
- Source: results/raw/reward_learning + training_logs/derived
- Note: Controlled RQ4 checkpoints are separate from Cycle-3 *_c3.pt

| Model | Seed | Stage/tag | Feasibility | Mean h | Protocol |
|---|---|---|---|---|---|
| History Transformer Routing (H2) | 42 | bc_only_rq4 | 20/20 | 22.5423 | official n10 construction-only |
| History Transformer Routing (H2) | 42 | rl_from_bc_rq4 | 20/20 | 23.4298 | official n10 construction-only |
| History Transformer Routing (H2) | 43 | bc_only_rq4 | 20/20 | 21.4002 | official n10 construction-only |
| History Transformer Routing (H2) | 43 | rl_from_bc_rq4 | 20/20 | 21.1998 | official n10 construction-only |
| History Transformer Routing (H2) | 44 | bc_only_rq4 | 20/20 | 22.5303 | official n10 construction-only |
| History Transformer Routing (H2) | 44 | rl_from_bc_rq4 | 20/20 | 24.0106 | official n10 construction-only |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | 42 | bc_only_rq4 | 20/20 | 22.6227 | official n10 construction-only |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | 42 | rl_from_bc_rq4 | 20/20 | 22.0479 | official n10 construction-only |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | 43 | bc_only_rq4 | 20/20 | 22.6678 | official n10 construction-only |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | 43 | rl_from_bc_rq4 | 20/20 | 21.4805 | official n10 construction-only |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | 44 | bc_only_rq4 | 20/20 | 22.1390 | official n10 construction-only |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | 44 | rl_from_bc_rq4 | 20/20 | 21.3808 | official n10 construction-only |
| State-Only Hierarchical Pointer (H1) | 42-44 | c3_log_bc_best | synth val | 15.5464 | historical Cycle-3 log (not controlled ckpt) |
| State-Only Hierarchical Pointer (H1) | 42-44 | c3_log_rl_end | synth val | 17.8534 | RL improved seeds: 0/3 |
| History Transformer Routing (H2) | 42-44 | c3_log_bc_best | synth val | 15.4235 | historical Cycle-3 log (not controlled ckpt) |
| History Transformer Routing (H2) | 42-44 | c3_log_rl_end | synth val | 22.0845 | RL improved seeds: 1/3 |
| DeepSets Fleet Routing (H3) | 42-44 | c3_log_bc_best | synth val | 15.6168 | historical Cycle-3 log (not controlled ckpt) |
| DeepSets Fleet Routing (H3) | 42-44 | c3_log_rl_end | synth val | 22.0632 | RL improved seeds: 0/3 |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | 42-44 | c3_log_bc_best | synth val | 15.4325 | historical Cycle-3 log (not controlled ckpt) |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | 42-44 | c3_log_rl_end | synth val | 26.4013 | RL improved seeds: 0/3 |

## Interpretation

Do not claim RL improves H4 unless controlled official/synth comparisons show it. Historical logs alone are not equivalent to saved BC-only vs RL checkpoints.
