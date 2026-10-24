# Targeted BC to RL improvement study

Reviewed parent: `f42a5b610ef50e5587a48e4432ee851e861741ec`.

The locked official evaluation is unchanged. This study does not replace `results/raw/bc_rl_full_v1/eval_stochastic/official_samples.csv` and does not rescore the Montoya benchmarks. The method stays behavior cloning, then REINFORCE, then neural route construction, then exact nonlinear charging. No PPO, actor–critic, ruin-and-repair, local search at inference, or new architecture was added.

The predeclared deployment rule was: a setting is promising only if at least 4 of 5 seeds have a negative RL−BC duration for the returned route and the paired Student-t 95% interval lies entirely below zero. Five seeds cannot produce a two-sided sign-test p-value below 0.0625. Intervals use the Student-t critical value 2.776 on 4 degrees of freedom. Every validation number below uses the 100 teacher-feasible instances in `results/raw/bc_rl_full_v1/teacher_cache_n10_train500_val100.pkl`. Those instances are the first 100 seeds at or after 11,200,000 whose teacher route is feasible. They are the split used to select the original checkpoints. They are not a fresh confirmation set.

## Answers

1. **Returned routes at a matched inference budget.** No. On the original Route Transformer checkpoints, one greedy construction plus three samples at temperature 1 gives RL−BC = −0.014 h, interval [−0.169, +0.140], on 2 of 5 seeds (t p = 0.808, sign p = 1.0, Wilcoxon p = 1.0). Temperature 0.8 gives −0.017 h, interval [−0.138, +0.103], on 3 of 5 seeds. The greedy-rollout checkpoints also miss the rule: Route Transformer −0.070 h, interval [−0.202, +0.063], 4 of 5 seeds; Route Transformer + DeepSets −0.036 h, interval [−0.162, +0.089], 4 of 5 seeds.
2. **Unseen geometric and charging distributions.** Not tested again. The earlier slow-charger holdout remains the recorded negative result. The distribution audit did not justify the one permitted mixed-distribution retraining, so no new generalization experiment was run.
3. **Does the policy benefit from nonlinear-charging information beyond the solver?** Not demonstrated. The feature diagnostic is in `docs/NONLINEAR_FEATURE_ABLATION.md`. No feature ablation was trained.
4. **Travel versus charging.** The returned-route differences are too small, and their intervals include zero, to assign a share of an improvement. The effect that is present is the expected duration of one sample. For the original checkpoints at temperature 1, that mean falls by 0.461 h. Travel accounts for −0.456 h, interval [−0.620, −0.292]. Charging accounts for −0.005 h, interval [−0.027, +0.018]. Service is 5.000 h on both sides.
5. **Replication and tests.** The returned-route comparisons do not meet the predeclared rule. The sample-mean comparisons that have an interval entirely below zero still have exact sign and Wilcoxon p = 0.0625. No fresh confirmation set was opened, because the validation rule failed.

## Phase 1 — greedy-anchored inference, no retraining

Protocol: `results/raw/bc_rl_improvement_v1/teacher_validation/frozen_protocol.json`. Raw rows: `evaluations.csv`, 8,000 constructions, elapsed 341.959 s. Checkpoints: `results/raw/bc_rl_full_v1/runs/H2_seed{42–46}/{bc_best,rl_best}.pt`. One greedy decode uses unscaled logits. Three samples use seeds `940000 + index*16 + trial`. Temperature rescales only the finite masked route and customer logits. Both temperatures were specified before the run. Tables: `results/tables/final/greedy_anchored_by_seed.csv` and `greedy_anchored_paired.csv`. Figure: `results/final/figures/fig_greedy_anchored_deltas.png`.

Every construction on this split is feasible: 1,000 of 1,000 greedy rows and 3,000 of 3,000 sample rows at each temperature. Greedy rows at temperature 0.8 match temperature 1.0, which checks that greedy decoding ignores temperature. Seed 42 behavior-cloning greedy duration is 21.4257 h, the same value recorded in the original training log.

Seed-mean durations at temperature 1.0, in hours:

| Protocol | BC | RL | RL − BC | 95% interval | Seeds shorter | Sign p | Wilcoxon p | t p |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Greedy returned route | 21.475 | 21.516 | +0.040 | [−0.109, +0.190] | 3/5 | 1.0 | 1.0 | 0.494 |
| Greedy + 3 samples, returned route | 21.210 | 21.196 | −0.014 | [−0.169, +0.140] | 2/5 | 1.0 | 1.0 | 0.808 |
| Mean of the 3 samples | 25.403 | 24.942 | −0.461 | [−0.629, −0.293] | 5/5 | 0.0625 | 0.0625 | 0.0016 |

Temperature 1.0 returned-route seed deltas, RL − BC, are +0.009, +0.116, +0.084, −0.177, and −0.105 h. Temperature 0.8 returned-route deltas are −0.023, −0.010, +0.001, −0.163, and +0.108 h. Its mean is −0.017 h, interval [−0.138, +0.103], 3 of 5 seeds, sign p = 1.0, t p = 0.708. The temperature-0.8 sample mean is −0.349 h, interval [−0.630, −0.068], 5 of 5 seeds, sign p = 0.0625.

The returned route is the shortest feasible candidate, with greedy ordered first in a tie. Its cost is the sum of all four decodes. At temperature 1.0 the seed-mean cost is about 0.163 s and 201 solver executions for behavior cloning, and 0.169 s and 204 executions for REINFORCE. One greedy decode costs about 0.040 s and 54.8 solver executions (`oracle_calls` equals `oracle_solver_executions` on the behavior-cloning greedy rows). The four-decode budgets match. Selecting the minimum makes both methods shorter than their own greedy output. The RL−BC difference of that minimum is the deployment comparison, and its interval includes zero.

For the temperature-1.0 returned route, travel changes by −0.033 h, interval [−0.163, +0.097], and charging changes by +0.019 h, interval [−0.012, +0.050]. For the mean of three samples, travel changes by −0.456 h, interval [−0.620, −0.292], and charging by −0.005 h, interval [−0.027, +0.018].

Neither temperature met the rule. No temperature was frozen. Official instances were not decoded under this protocol.

An earlier file, `results/raw/bc_rl_improvement_v1/evaluations.csv`, decoded seeds 11,200,000 through 11,200,099 in order. Twenty-two of those seeds are absent from the teacher cache, and every checkpoint left the same 22 greedy rows infeasible. That file is retained and is not the decision set.

## Phase 2 — detached greedy-rollout REINFORCE

The inference result missed the deployment rule, so the prepared experiment in `scripts/experiments/run_final_rl.py` was run once. It starts from the immutable behavior-cloning checkpoints, trains H2 and H4, seeds 42–46, for 300 updates, with shuffled instance order, four sampled trajectories, baseline `greedy`, learning rate 2e-5, normalized nonlinear duration reward, and entropy and KL coefficients of zero. Output: `results/raw/bc_rl_final_v1/`. All 10 runs completed 300 updates. Failures: 0. Official evaluation was not started.

Audit before training:

- The greedy decode runs under `model.eval()` and `torch.no_grad()`.
- `reward_value` returns a Python float, and that same float is the baseline of every rollout in the batch.
- `reinforce_loss` builds the advantage as a tensor and calls `.detach()` before multiplying by the log-probability.
- The trajectory log-probability is the sum of the route-choice and customer-choice log-probabilities.

A rollout baseline reduces the variance of the policy gradient. It does not guarantee that greedy duration will fall. Checkpoint selection remains the existing rule: higher feasibility, then lower greedy penalized validation duration. The selected greedy durations reproduce the training comparison in `results/raw/bc_rl_final_v1/validation_comparison.csv`.

Greedy validation, RL − BC, in hours:

| Model | Deltas | Mean | 95% interval | Seeds shorter | t p | Sign p | Wilcoxon p |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| H2 | −0.017, −0.095, +0.125, −0.108, −0.187 | −0.057 | [−0.203, +0.090] | 4/5 | 0.345 | 0.375 | 0.438 |
| H4 | −0.019, −0.050, −0.010, +0.053, +0.152 | +0.025 | [−0.075, +0.125] | 3/5 | 0.524 | 1.0 | 0.812 |

H2 seed 44 and H4 seeds 45 and 46 have a higher greedy validation duration after REINFORCE. The sum of `greedy_oracle_calls` across the ten reward logs is 329,907. Eight runs record 33,000 calls. H2 seed 46 records 32,983 and H4 seed 42 records 32,924.

The same teacher-feasible instances were then scored with one greedy decode plus three samples at temperature 1. Protocol: `results/raw/bc_rl_final_v1/anchored_validation/frozen_protocol.json`. Elapsed 350.469 s. Every returned route is feasible. Table: `results/tables/final/greedy_rollout_anchored_paired.csv`.

| Model | Returned route | Greedy | Mean of 3 samples |
| --- | --- | --- | --- |
| H2 | −0.070 h [−0.202, +0.063], 4/5, t p = 0.218, sign p = 0.375 | −0.057 h [−0.203, +0.090], 4/5 | −0.193 h [−0.363, −0.023], 5/5, sign p = 0.0625 |
| H4 | −0.036 h [−0.162, +0.089], 4/5, t p = 0.467, sign p = 0.375 | +0.025 h [−0.075, +0.125], 3/5 | −0.466 h [−0.927, −0.004], 5/5, sign p = 0.0625 |

H2 returned-route seed deltas are −0.018, −0.077, +0.081, −0.190, and −0.145 h. The four-decode cost is about 0.168 s and 202 solver executions for H2, and 0.179 s and 202 executions for H4, averaged across both roles. On the H2 returned route, travel changes by −0.066 h, interval [−0.183, +0.051], and charging by −0.004 h, interval [−0.031, +0.024]. On the H4 returned route, travel changes by −0.067 h, interval [−0.199, +0.064], and charging rises by +0.031 h, interval [+0.001, +0.061], on 0 of 5 seeds. The H4 charging increase is the one component interval that excludes zero, and the full-duration interval still includes zero.

The sample-mean intervals that lie below zero are expected draws on the checkpoint-selection split. They are smaller than a claim about a returned route, and they were not taken to a new confirmation set.

## Phase 3 — distribution audit, no new training

`scripts/reporting/compare_instance_distributions.py` writes `results/tables/final/distribution_coverage.csv`. The comparison is structural. Routes were not solved, so charging duration and the margin to the 10 h limit were not measured.

| Group | n | Slow-station fraction | Slow-only instances | Mean depot–customer distance | Max depot energy / battery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher-cache training | 500 | 0.392 | 0.144 | 61.087 km | 0.726 |
| Teacher-cache validation | 100 | 0.357 | 0.140 | 65.009 km | 0.768 |
| Slow-charger holdout | 20 | 1.000 | 1.000 | 36.034 km | 0.442 |
| Official n=10 | 20 | 0.467 | 0.000 | 69.674 km | 0.751 |

Slow chargers are already present in the training cache. Official n=10 is close to that cache in distance and energy. The holdout is slow-only and also geographically shorter, with a lower depot-energy ratio. It is a joint shift, and it is not a harder energy regime than training. The one permitted H2 mixed-distribution run was not started. Raising the slow-only rate would be a different experiment, and this audit does not show that the original training omitted slow charging. The existing holdout result stands: the sampled-mean reduction did not reappear there.

## Phase 4 — charging features

The diagnostic and the decision not to retrain are in `docs/NONLINEAR_FEATURE_ABLATION.md`. The solver’s use of nonlinear physics remains the fixed-route result already in the paper: 20 of 20 true-curve plans stay feasible at n=40, against 11 of 20 for the two linear planners. The policy’s response to curvature, beyond the station-type label, was not shown.

## What was not done

- No temperature was frozen, and no official Montoya confirmation was run.
- No additional training seeds were added.
- The mixed-distribution REINFORCE run was not started.
- The matched charging-feature ablation was not trained.
- Historical checkpoints and the locked official sample file were not overwritten. The new weights are in `results/raw/bc_rl_final_v1/runs/`.
