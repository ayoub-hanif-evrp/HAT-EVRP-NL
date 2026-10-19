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

On this construction-only slice, RL is slightly better only for Route Transformer. The other three models are worse after RL. A deployed checkpoint that keeps BC is not an RL improvement. P1, P2, and the larger sizes are in the per-seed CSVs and are not summarized here.

Targeted ablations (multi-target BC, reward, baseline, decoder weights) were not part of this matrix. Historical Cycle-3 files were not overwritten. Figures 12–21 and the matching manuscript captions now use these CSVs. Figures 1–11 remain the charging and layout study.
