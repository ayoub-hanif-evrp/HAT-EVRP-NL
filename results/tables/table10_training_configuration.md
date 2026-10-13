# Training configuration for Cycle-3 H1–H4 checkpoints

- Objective convention: Synthetic validation full duration for checkpoint selection
- Units: hours / steps
- Benchmark / data: n=10 synthetic train/val
- Instance / seed count: 120 train / 12 val per seed
- Methods / checkpoints: H1–H4 Cycle-3
- Protocol: BC then standard REINFORCE (pomo_k=0); generator v2 defaults
- Seeds: 42–44
- Source raw: `results/archive/reports/notes.md + train.py defaults + training_*_c3.csv`

| Display name | Key | History | DeepSets | Hidden | Layers | Heads | BC steps | RL type | RL steps | Train inst. | Val inst. | Seeds | Optimizer | Checkpoint rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| State-Only Hierarchical Pointer (H1) | H1 | no | no | 128 | 2 | 4 | 400 BC | standard REINFORCE | 40 | 120 synth n=10 | 12 synth n=10 | 42,43,44 | Adam lr=default | best (feas, −val obj) over logged steps |
| History Transformer Routing (H2) | H2 | yes | no | 128 | 2 | 4 | 400 BC | standard REINFORCE | 40 | 120 synth n=10 | 12 synth n=10 | 42,43,44 | Adam lr=default | best (feas, −val obj) over logged steps |
| DeepSets Fleet Routing (H3) | H3 | no | yes | 128 | 2 | 4 | 400 BC | standard REINFORCE | 40 | 120 synth n=10 | 12 synth n=10 | 42,43,44 | Adam lr=default | best (feas, −val obj) over logged steps |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | H4 | yes | yes | 128 | 2 | 4 | 400 BC | standard REINFORCE | 40 | 120 synth n=10 | 12 synth n=10 | 42,43,44 | Adam lr=default | best (feas, −val obj) over logged steps |

## Interpretation

Insertion/POMO/repair flags were **not** used for Cycle-3 primary checkpoints. Post-Cycle-3 `perf_*` runs are separate warm-start experiments.
