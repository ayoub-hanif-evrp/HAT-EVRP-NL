# Remaining limitations after the coverage and baseline corrections

The original 20-run matrix in `results/raw/bc_rl_full_v1` is unchanged. The corrected RL runs are in `results/raw/bc_rl_shuffle_v1`. Classical execution-matched rows are in `results/raw/classical_matched_v1/official.csv`.

## What changed in code

REINFORCE can draw a deterministic shuffle that finishes all 500 training instances before repeating. The old cyclic index remains available and is what the original 20 runs used. A run can start from an immutable `bc_best.pt` and execute only the RL phase. The full-matrix runner now requires 1,000 BC updates, 300 RL updates, readable checkpoint payloads, 480 evaluation rows, and P2 accounting `construction_plus_hybrid` before it treats a run as complete. The P2 recount itself is unchanged.

## Corrected RL, validation

Frozen question, written before the runs: does shuffled RL lower synthetic validation duration relative to the original cyclic RL, for Route Transformer and RT + DeepSets, seeds 42–46, from the saved behavior-cloning weights, with 300 updates and the same normalized reward?

It does not. Paired difference (shuffled − original), hours:

| Model | Mean | 95% interval | Shuffled lower |
|---|---:|---|---:|
| Route Transformer | +0.035 | [−0.046, +0.116] | 1/5 |
| RT + DeepSets | +0.044 | [−0.033, +0.121] | 2/5 |

Shuffled RL beats behavior cloning on 1/5 Route Transformer seeds and 0/5 RT + DeepSets seeds. For RT + DeepSets the difference versus behavior cloning is +0.114 h, interval [+0.011, +0.218]. That second comparison was not the frozen hypothesis. It is reported because the corrected phase is worse than the checkpoint it started from on every RT + DeepSets seed. Source: `results/raw/bc_rl_shuffle_v1/validation_comparison.csv`.

Official P0/P1/P2 evaluation of these checkpoints was started only after that table existed. It is not used to decide whether the sampler worked.

## Equal-budget classical search

Deterministic greedy, then either multi-start local search or ruin-and-repair, was run on the same official instances. The execution cap is the mean neural solver-execution count of each protocol. Realized executions match those caps. Requests and cache hits are recorded separately and are larger than executions.

At the construction cap, greedy is feasible on 12/20 (n=10), 17/20 (n=20), 0/20 (n=40), and 0/20 (n=80). Route Transformer behavior cloning is feasible on all 20 instances at every size. On the n=10 instances both methods finish, durations are close (20.153 h versus 20.135 h). At the larger caps, where classical search also finishes every instance, Route Transformer behavior cloning remains lower by about 0.6–1.1 h at n=10, 2.7–3.5 h at n=20, 3.1 h at n=40, and 4.5 h at n=80. Classical ruin-and-repair does not beat greedy plus local search. Extra executions past the one-shot cap do not change the classical means at n=40 or n=80. Summary: `results/tables/classical_matched_v1_summary.csv`.

A separate arm stops at the neural mean runtime. At n=10 that time lets classical search execute more solves than the neural construction count, and ruin-and-repair matches or beats Route Transformer behavior cloning (21.749 h, 20.054 h, and 19.527 h versus 22.379 h, 20.997 h, and 19.974 h). At n=20, 40, and 80, Route Transformer remains lower by about 2.5–4.7 h. Both arms are in `results/tables/classical_matched_v1_summary.csv`.

## Statistics

Two primary comparisons were declared for the original matrix: Route Transformer, official n=10, construction, RL minus BC; and the same slice for RT + DeepSets. The first is −0.091 h, about 0.4% of the behavior-cloning mean, interval [−0.336, +0.154]. The second is +0.217 h, interval [−0.241, +0.676]. The other 46 BC-versus-RL cells are exploratory. Forty-seven of 48 intervals include zero. The one that does not is Fleet DeepSets, n=10, hybrid insertion, and it says RL is worse. All five seeds share one synthetic training set, so the intervals do not measure a new draw of the training data. P0, P1, and P2 on one size reuse the same instances and checkpoints.

## What is still open

- Official scores for the shuffled checkpoints are recorded (10 files, 4,800/4,800 feasible, P2 accounting included). They do not overturn the validation result above.
- Matched wall-clock classical search is recorded. It changes the n=10 reading: classical search is competitive when the budget is time rather than solver executions. At n=20 and above, Route Transformer behavior cloning remains lower.
- The manuscript now states the formulation, the two training losses, the oracle accounting, and the execution-matched protocol. It is still a draft: the related-work discussion is short of a full review, and the wall-clock arm is absent.
- REINFORCE does not improve greedy construction. It does improve the mean of four matched samples on official n=10 and n=20 for the Route Transformer and RT + DeepSets (`docs/FINAL_BC_RL_RESEARCH_REPORT.md`). The sampled log-ratio in `results/raw/bc_rl_kl_v1` has expected gradient zero. The distribution term in `results/raw/bc_rl_kl_v2` is a sampled-conditional KL, not the exact hierarchical KL, and it is not the locked comparison. DeepSets is not an established improvement over the Route Transformer. Classical search is supplementary and is not the RL result.

The test suite against this revision passed 95 tests (`results/raw/bc_rl_current_test_transcript.txt`).
