# BC→RL full experiment

Status: **FULLY TRAINED and FULLY EVALUATED** for the predeclared 20-run matrix. Exit code 0. Failures: 0. Wall clock about 9.5 hours.

Manifest: `results/raw/bc_rl_full_v1/experiment_manifest.csv`.

Every run completed 1,000 behavior-cloning updates and 300 reinforcement-learning updates on the shared synthetic set (n=10, 500 train, 100 validation, `data_seed=1000`, 4 leave-one-out rollouts). Official evaluation CSVs cover construction (P0), fair local search (P1), and hybrid insertion (P2) for n=10, 20, 40, and the available n=80 instances. Checkpoints remain separate: `bc_best.pt`, `rl_best.pt`, `deployed_best.pt`.

Synthetic-validation selections (RL better than BC, then deployed):

| Model | Seeds where RL won validation |
|---|---|
| State Pointer | 2/5 |
| Route Transformer | 3/5 |
| Fleet DeepSets | 1/5 |
| Route Transformer + DeepSets | 1/5 |

Official n=10 construction-only, mean of the five seed means, 100/100 feasible in every cell:

| Model | BC (h) | RL (h) |
|---|---:|---:|
| State Pointer | 22.253 | 22.655 |
| Route Transformer | 22.379 | 22.287 |
| Fleet DeepSets | 22.060 | 22.700 |
| Route Transformer + DeepSets | 22.065 | 22.282 |

On this construction-only slice, the Route Transformer mean is 0.091 h lower after RL and the other three means are higher. The paired Student-t 95% interval on the five seed differences for that Route Transformer change is [−0.336, +0.154] and includes zero. Under hybrid insertion the same comparison is +0.038 h, with RL better on 1/5 seeds.

Every official row is feasible (9,600/9,600). The paired summary for all four sizes and P0, P1, and P2 is `results/tables/bc_rl_full_v1_paired.csv`. Forty-seven of 48 intervals include zero. The exception is Fleet DeepSets at n=10 hybrid insertion: RL − BC = +0.192 h, interval [+0.022, +0.362], RL better on 0/5 seeds. A deployed checkpoint that keeps BC is not an RL improvement. These comparisons do not support a general RL benefit.

P2 oracle executions, requests, cache hits, and wall-clock time now cover construction plus classical ruin plus neural repair. Recomputing all 3,200 P2 rows changed no objective (0 mismatches). At n=10 the mean solver-execution counts are about 55 for construction, 162 for one-shot local search, and 418 for hybrid insertion. Every P2 count is above the construction count on the same instance. Figure 17 uses these corrected counts.

Targeted ablations (multi-target BC, reward, baseline, decoder weights) were not part of this matrix. Historical Cycle-3 files were not overwritten. Figures 12–21 and the matching manuscript captions now use these CSVs. Figures 1–11 remain the charging and layout study.
