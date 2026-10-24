# Final experiment execution report

Parent commit: `789af01b5c338fe814c470a46c31e78e070f39e6`.

The locked official Montoya evaluation was not rerun and was not overwritten. Historical teacher-cache validation remains historical evidence. New decisions use the splits in `results/raw/final_autonomous_study/splits/`.

## Answers

**Did RL produce better returned EV routes than BC under matched computational budgets?** No. On the unfiltered development and confirmation sets, every returned-route interval for the original REINFORCE checkpoint, the greedy-baseline checkpoint, and the best-of-K checkpoint includes zero. Feasibility rates match the behavior-cloning checkpoint on every split, so the penalized comparison is not an artifact of unequal failure rates.

**Did an improvement survive independent confirmation and distribution shifts?** The development rule was not met, and the locked confirmation set agrees. One shift, the Montoya-like `v2c` sample, has a returned-route interval below zero for the original REINFORCE checkpoint. The slow-only and geographic shifts do not. That is one favorable synthetic family, not generalized robustness.

**Did nonlinear curve information contribute to the learned policy beyond exact charging feasibility?** Not in the tests that can speak to that question. The matched feature ablation changes almost no routes and does not improve duration. The established result remains exact nonlinear charging by `frvcpy`.

## Evaluation protocol

Instances were drawn from documented seeds and kept whether or not a teacher route existed. Hashes were written before decoding: `results/raw/final_autonomous_study/splits/hashes.csv`, 368 rows. Infeasible decodes use penalized duration `n × T_max = 100` h. The primary metric is the penalized duration of the shortest feasible route among one greedy decode and three samples. Sample seeds are fixed in the split protocol. Temperature is 1.

| Split | Role | Seeds | n |
| --- | --- | --- | ---: |
| `dev_v2` | Development validation | 31,000,000–31,000,099 | 100 |
| `confirm_v2` | Locked confirmation | 32,000,000–32,000,099 | 100 |
| `shift_v2c` | Montoya-like geometry | 33,000,000–33,000,039 | 40 |
| `shift_slow` | Slow-only, clustered, p-median | 34,000,000–34,000,039 | 40 |
| `shift_geo` | Clustered, offset depot, elongated | 35,000,000–35,000,039 | 40 |
| `curve_probe` | Same geometry, perturbed curve slopes | 36,000,000–36,000,023 | 24 pairs |

About one quarter of the unfiltered v2 development instances have no feasible returned route. The earlier teacher-cache validation had removed those seeds. Returned-route feasible rates below are identical for behavior cloning and each RL checkpoint.

Four decodes cost about 0.17–0.21 s and about 200–206 solver executions per instance. The three existing checkpoints match each other to within a few executions.

## Existing checkpoints

Paired RL − BC penalized returned-route duration. Tables: `results/tables/final/unbiased_eval_paired.csv` and `unbiased_eval_by_seed.csv`. Figure: `results/final/figures/fig_unbiased_returned_routes.png`. Raw rows: `results/raw/final_autonomous_study/unbiased_eval/evaluations.csv`, elapsed 876.7 s.

| Split | Candidate | Mean | 95% interval | Seeds shorter | Sign p | Wilcoxon p | Feasible rate |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Development | Original RL | −0.011 h | [−0.131, +0.110] | 2/5 | 1.0 | 1.0 | 0.76 |
| Development | Greedy-baseline RL | −0.027 h | [−0.102, +0.048] | 2/5 | 1.0 | 0.812 | 0.76 |
| Confirmation | Original RL | −0.049 h | [−0.131, +0.033] | 4/5 | 0.375 | 0.188 | 0.75 |
| Confirmation | Greedy-baseline RL | −0.026 h | [−0.141, +0.088] | 2/5 | 1.0 | 1.0 | 0.75 |
| Montoya-like v2c | Original RL | −0.175 h | [−0.251, −0.099] | 5/5 | 0.0625 | 0.0625 | 0.60 |
| Montoya-like v2c | Greedy-baseline RL | −0.078 h | [−0.154, −0.003] | 5/5 | 0.0625 | 0.0625 | 0.60 |
| Slow only | Original RL | +0.034 h | [−0.143, +0.211] | 2/5 | 1.0 | 0.812 | 0.975 |
| Slow only | Greedy-baseline RL | +0.003 h | [−0.212, +0.219] | 2/5 | 1.0 | 0.812 | 0.975 |
| Geography | Original RL | −0.162 h | [−0.492, +0.169] | 2/5 | 1.0 | 0.812 | 0.75 |
| Geography | Greedy-baseline RL | −0.102 h | [−0.384, +0.180] | 2/5 | 1.0 | 0.812 | 0.75 |

On the v2c shift, behavior cloning and original REINFORCE fail the same 200 of 200 seed-instance pairs. The returned-route difference is therefore not a change in which instances are feasible. Travel on the feasible returned routes is −0.249 h, interval [−0.355, −0.144]. Charging is −0.042 h, interval [−0.141, +0.057]. The greedy-only interval on that split includes zero. Five of five seeds cannot produce a two-sided sign-test p-value below 0.0625.

The slow-only and geographic shifts do not reproduce a returned-route gain. Mixed-distribution retraining was not run. Slow chargers are already present in the original training cache, and the new slow split is not a harder energy regime that the audit isolates from geography. One favorable `v2c` cell is not a reason to start another training distribution.

## Objective-aligned REINFORCE

Protocol: `results/raw/final_autonomous_study/objective_aligned/frozen_protocol.json`. H2, seeds 42–46, from the saved behavior-cloning checkpoints, 300 updates, learning rate \(2\times 10^{-5}\), three samples, detached greedy candidate. All five runs completed. Failures: 0. Elapsed about 31 minutes.

The loss is \(-(R_\mathrm{best}-b)\sum_k \log\pi(\tau_k)\). \(R_\mathrm{best}\) is the best normalized reward among the greedy decode and the three samples. \(b\) is the greedy reward. The advantage is detached and nonnegative. Every sample log-probability receives that coefficient, including samples that did not win. The greedy decode runs under `torch.no_grad`, so the estimator omits the greedy policy's dependence on the parameters. It is the score function for the sampling distribution with the greedy route held fixed. Unit tests check that both sample log-probabilities receive the same gradient coefficient.

Checkpoint selection used only `dev_v2`, with the anchored penalized duration. The external rescoring of the saved checkpoints reproduces that selection: −0.053 h, interval [−0.159, +0.054], 4 of 5 seeds, t p = 0.24. The rule required the interval to lie entirely below zero. It does not. Seed deltas are −0.009, −0.054, −0.005, +0.005, and −0.201 h.

| Split | Mean | 95% interval | Seeds shorter |
| --- | ---: | --- | ---: |
| Development | −0.053 h | [−0.159, +0.054] | 4/5 |
| Confirmation | −0.037 h | [−0.157, +0.083] | 3/5 |
| Montoya-like v2c | −0.012 h | [−0.172, +0.147] | 3/5 |
| Slow only | −0.040 h | [−0.199, +0.119] | 2/5 |
| Geography | −0.120 h | [−0.381, +0.142] | 3/5 |

Training solver calls were 606,249, against 766,616 for the five H2 greedy-baseline runs. The difference matches the decode count: three samples plus one greedy decode, versus four samples plus one greedy decode. The best-of-K run was not given a larger solver budget. No further objective was trained.

## Curve-shape tests

The inference probe keeps labels and coordinates fixed and perturbs breakpoint slopes so that charging remains slower at higher SOC. Official Montoya curves are unchanged. Greedy routes from the original checkpoints change on 13 of 120 behavior-cloning pairs and 15 of 120 REINFORCE pairs. Feasibility does not flip. Mean penalized duration moves by +0.130 h and +0.142 h. A route change at inference can come from the charging solver's feasibility mask or from route history. It does not identify the static curve features.

The matched ablation trains both arms from scratch on the same 40 curve-varied training instances and 20 validation instances, H2, seeds 42–46, 40 behavior-cloning steps and 40 REINFORCE steps, leave-one-out baseline, learning rate \(2\times 10^{-5}\). Eighteen teacher seeds were skipped because the teacher route was infeasible; those skips are stored in the cache and were not used to choose the probe. The probe itself was not used for selection. One arm receives the usual four curve features. The other zeroes those features and keeps the station-type indicators. Capacity, data, seeds, and the optimizer match.

On the 24 probe pairs, greedy features-minus-masked duration is −0.001 h for behavior cloning and +0.000 h for REINFORCE. Every interval includes zero. Routes differ on 4 of 120 pairs in each role. Feasibility is 85 of 120 for both arms. This budget is shorter than the main study. Within that matched comparison, explicit curve features do not improve returned routes or change routing in a measurable way.

## Physically coverable instances

The penalized means on the unfiltered sets are high because an infeasible construction is scored at \(n T_{\max}=100\) h. On development seed 42, behavior cloning, the original REINFORCE checkpoint, and the greedy-baseline checkpoint fail the same 24 of 100 instances. On confirmation they fail the same 25 of 100. Those sets are identical across seeds 42–46.

Every failed instance contains at least one customer with no feasible dedicated route under `FixedRouteOracle`. Every instance that the neural constructor solves has a feasible dedicated route for all ten customers. A strong randomized-greedy search (8 restarts, 3 local-search passes, budget 2500, strong operators) solves 0 of those 24 development instances and 0 of those 25 confirmation instances. The shared penalty is therefore an instance that cannot cover every customer, not a route the REINFORCE update missed.

Restricting the returned-route comparison to instances both methods solve removes that shared penalty. Original REINFORCE minus behavior cloning is −0.014 h on 76 development instances, interval [−0.172, +0.145], on 2 of 5 seeds (behavior cloning 18.769 h, REINFORCE 18.755 h). On 75 confirmation instances it is −0.066 h, interval [−0.175, +0.044], on 4 of 5 seeds (20.288 h and 20.222 h). Both intervals include zero. The v2c cell remains the only interval entirely below zero, −0.291 h on 24 solvable instances, interval [−0.418, −0.165], sign-test p = 0.0625. Slow and geographic shifts still include zero. The audit is `results/tables/final/singleton_feasibility_audit.csv`. The paired table is `results/tables/final/solvable_subset_paired.csv`.

## Experiment ledger

| Experiment | Status | Result |
| --- | --- | --- |
| Unbiased evaluation of original BC, original RL, and greedy-baseline RL | PASS | Completed on all five decision splits. No reliable returned-route gain on development or confirmation. |
| Objective-aligned best-of-K REINFORCE | NEGATIVE | Five seeds completed. Development interval includes zero. Line stopped. |
| Mixed-distribution REINFORCE | NEGATIVE | Not run. The shift results and the existing coverage audit did not justify it. |
| Curve-probe inference | PASS | Completed. Routes change on about 11–12% of pairs. Feasibility does not. |
| Matched curve-feature ablation | NEGATIVE | Ten runs completed. No duration or routing benefit from the curve features. |
| Official Montoya rescoring | NEGATIVE | Not run. Prior exposure stands, and no new candidate met the development rule. |
| Singleton coverability audit and solvable-subset rescoring | PASS | Strong search solves 0 of 49 uncovered development and confirmation instances. Returned-route intervals on the coverable subset still include zero. |

Raw outputs live under `results/raw/final_autonomous_study/`. The manuscript limitations section records the same negative conclusion without replacing the locked official numbers.
