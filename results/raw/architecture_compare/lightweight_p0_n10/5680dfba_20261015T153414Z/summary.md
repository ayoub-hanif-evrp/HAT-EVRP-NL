# Lightweight P0 n=10 architecture compare

Protocol: construction-only (greedy decode, `cost_lambda=0`), official n=10, 20/20 instances, eval seed 42.

| Method (display) | Internal key | Feasible | Mean Z_full (h) | Checkpoint |
|---|---|---:|---:|---|
| State-only Pointer | H1 | 20/20 | 22.4990 | `H1_n10_seed42_c3.pt` |
| History Transformer | H2 | 20/20 | 21.8444 | `H2_n10_seed42_c3.pt` |
| DeepSets Fleet Model | H3 | 20/20 | 22.6678 | `H3_n10_seed42_c3.pt` |
| History Transformer + DeepSets | H4 | 20/20 | 21.8444 | `H4_n10_seed42_c3.pt` |
| History + Open-route DeepSets | H4_open | 20/20 | 21.8001 | `H4_open_n10_seed42_perf_v2mt_s.pt` |

## Interpretation (honest)

- On this **single training seed (42)** and construction-only protocol, History Transformer and History+DeepSets yield **identical** mean objectives (21.8444 h) — consistent with prior notes that DeepSets does not change routing on this seed under construction-only.
- `H4_open` is slightly lower (21.8001 h) but uses an **unmatched** `perf_v2mt_s` checkpoint (not Cycle-3 matched training). **Do not treat as a controlled open-route DeepSets win.**
- Prefix-only history ablation still requires a new encoder code path + multi-seed training (deferred; see `experiment_manifest.yaml`).
