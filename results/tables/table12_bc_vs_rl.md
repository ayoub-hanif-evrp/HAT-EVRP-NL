# Behavior cloning vs REINFORCE on synthetic validation (Cycle-3)

- Objective convention: Synthetic validation full duration (h)
- Units: hours
- Benchmark / data: n=10 synthetic
- Instance / seed count: 12 validation instances × 3 seeds
- Methods / checkpoints: H1–H4
- Protocol: Log-based: best BC step vs RL endpoint; checkpoint keeps best overall
- Seeds: 42–44
- Source raw: `results/raw/training_logs/derived/bc_vs_rl_aggregate_c3.csv`

| Method | Best BC val h | RL endpoint val h | Δ (RL−BC) | Seeds RL improved | Likely deployed stage | Sampled reward traces |
|---|---|---|---|---|---|---|
| State-Only Hierarchical Pointer (H1) | 15.5464 ± 1.7422 | 17.8534 ± 3.7682 | +2.3070 ± 2.5763 | 0/3 | 3/3 BC; 0/3 RL | unavailable (historical) |
| History Transformer Routing (H2) | 15.4235 ± 1.6718 | 22.0845 ± 12.4546 | +6.6610 ± 11.1450 | 1/3 | 2/3 BC; 1/3 RL | unavailable (historical) |
| DeepSets Fleet Routing (H3) | 15.6168 ± 1.8694 | 22.0632 ± 7.7360 | +6.4463 ± 9.4512 | 0/3 | 3/3 BC; 0/3 RL | unavailable (historical) |
| History Transformer + DeepSets (HAT-EVRP-NL) (H4) | 15.4325 ± 1.7459 | 26.4013 ± 7.5113 | +10.9688 ± 8.9199 | 0/3 | 3/3 BC; 0/3 RL | unavailable (historical) |

## Interpretation

Across three Cycle-3 seeds, RL **endpoint** validation beats best-BC on only **1/3 seeds for H2** and **0/3 for H4** (and H1/H3). Mean RL endpoints are worse because several seeds collapse. Checkpoint selection still retains BC weights when RL worsens validation — so deployed H4 (and often H1/H3) are BC-selected. Do **not** claim RL improves HAT-EVRP-NL. Separate BC-only `.pt` files were not saved; official Montoya BC-only vs RL weight evaluation remains unavailable without a controlled retrain.
