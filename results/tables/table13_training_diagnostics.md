# Training diagnostics (Cycle-3 History / History+DeepSets per seed)

- Objective convention: Synthetic validation + logged losses/entropy
- Units: hours / nats
- Benchmark / data: n=10 synthetic
- Instance / seed count: 12 val instances
- Methods / checkpoints: History Transformer / History Transformer + DeepSets
- Protocol: Cycle-3 logs; RL only logged at endpoint (eval_every=40, rl_steps=40)
- Seeds: 42–44
- Source raw: `results/raw/training_logs/derived/bc_vs_rl_per_seed_c3.csv`

| Method | Seed | BC-end entropy | RL-end entropy | RL-end PG loss | BC-best val h | RL-end val h | Deployed stage (log) | RL stability |
|---|---|---|---|---|---|---|---|---|
| History Trans. | 42 | 0.660 | 0.593 | -75.27 | 15.7137 | 15.4834 | rl | stable/mild |
| History Trans. | 43 | 0.708 | 0.682 | -36.62 | 13.6256 | 14.3199 | bc | stable/mild |
| History Trans. | 44 | 0.623 | 1.096 | 452.20 | 16.9311 | 36.4501 | bc | unstable |
| History + DeepSets | 42 | 0.625 | 1.074 | 536.77 | 15.7076 | 31.0154 | bc | unstable |
| History + DeepSets | 43 | 0.770 | 0.644 | 0.00 | 13.5654 | 30.4544 | bc | unstable |
| History + DeepSets | 44 | 0.644 | 0.757 | -546.91 | 17.0245 | 17.7341 | bc | stable/mild |

## Interpretation

Entropy and RL loss are diagnostics only. Sampled reward variance was not logged historically. History+DeepSets seed-42 RL endpoint 31.02 h vs BC-best ~15.71 h is a clear instability.
