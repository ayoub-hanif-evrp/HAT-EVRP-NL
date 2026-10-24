# REINFORCE root cause and the return-to-go correction

Starting commit `376cce83c383af00c80cbac3d7da68eb3c2c847f`. Historical checkpoints, official Montoya samples, and earlier result trees were left in place. The manuscript was not changed. The predeclared 1% rule on the returned route failed, so Montoya-like, slow-charger, and geographic shifts were not decoded and no further training configuration was run.

## Answers

1. **The original REINFORCE gradient is correct.** On a fully enumerated one-step and two-step categorical problem, the implemented loss matches the autograd of the expected normalized objective within `1e-6`. A central-difference check of that same expected duration matches autograd within `1e-8`. A constant baseline does not change the expected gradient. Minimizing the loss descends expected duration because the reward is minus duration. The original H2 leave-one-out logs have a nonzero gradient on all 1,500 updates (300 per seed).

2. **Best-of-K is zero when the greedy route wins.** In the five objective-aligned H2 logs, `grad_norm == 0` on 940 of 1,500 updates (62.7%). The logged advantage is missing on exactly those updates. `best_of_k_advantage` is `max(sample rewards) − greedy reward` and is defined to be zero when no sample is better, and the training step then skips the loss. Those zero advantages were left at zero.

3. **The bottleneck is the greedy ranking.** The original estimator already had a usable gradient, and greedy validation still did not keep improving. The guided decoder’s shorter routes diverge at early and middle decisions, especially second-best route choices, and those actions stay unlikely. Return-to-go raises the gradient on early decisions and still does not move the argmax. Five hundred coverable training instances do not remove that gap. Selecting on greedy plus three samples does not either.

4. **The correction is exact nonlinear return-to-go.** Each feasible decision is charged the change in total exact route duration. Those changes add up to `Z_full`. The reward scale is the existing normalized objective, `−Z_full / (n T_max)`. The baseline at each decision is the mean return-to-go of the other rollouts and is detached. Infeasible trajectories keep the existing terminal penalty on the last decision. This is the score-function estimator of that sum. It is the same physical objective.

5. **Corrected RL did not beat an equally trained behavior-cloning control on the returned route.** On the validation set used for selection, return-to-go minus behavior cloning is −0.021 h, interval [−0.158, +0.116], shorter on 4 of 5 seeds (−0.041%). The interval includes zero. The 1% rule required at least 4 of 5 seeds and an interval entirely below zero.

6. **The confirmation split does not show that improvement.** On 40 previously unused instances, return-to-go greedy minus behavior-cloning greedy is +0.023 h, interval [−0.066, +0.111], shorter on 3 of 5 seeds (+0.053%). The coverable subset is +0.031 h, interval [−0.091, +0.154]. Return-to-go and the original leave-one-out estimator agree to about 0.000 h on the greedy route. Shift splits and a larger replication were not run.

## 1. Why so many best-of-K updates are zero

Source: `results/raw/final_autonomous_study/objective_aligned/runs/H2_seed{42–46}/reward.csv`, summarized in `results/raw/reinforce_diagnosis_v1/gradient_audit.json`.

| Quantity | Value |
| --- | ---: |
| RL updates | 1,500 |
| Updates with `grad_norm == 0` | 940 (62.7%) |
| Updates with at least one sample reward above greedy | 560 (37.3%) |
| Updates whose mean sample reward exceeds greedy | 118 (7.9%) |
| Sampled routes in the logs (`n_rollouts` summed) | 9,000 |
| Lower bound on the share of routes that beat greedy | 560 / 9,000 = 6.2% |
| Mean absolute sample-minus-greedy reward on the 560 nonzero updates | 0.0253 (SD 0.0135) |
| Mean reward standard deviation | 0.0244 |
| Mean entropy | 0.758 (first 50 updates 0.712, last 50 updates 0.798) |
| Charging-solver executions | 606,249 |
| Nonzero-gradient updates per 1,000 solver executions | 0.924 |

The route-level share is a lower bound. The logs store one row per update, so an update that beats greedy contributes one winning route to the numerator even if several of its samples won. The exact per-route win rate was not stored. Zero gradient and a missing advantage coincide on every row.

The zero rate is present in every seed and in the last 100 updates as well as the first 100: 57–68% late, except seed 43, which falls from 71% to 47%. This is not an early-training transient. Entropy rises across the run, so the policy is not collapsing to a deterministic greedy action while the gradient disappears. The gradient disappears because the advantage definition is zero whenever greedy is best.

The original leave-one-out H2 logs, `results/raw/bc_rl_full_v1/runs/H2_seed{42–46}/reward.csv`, have 0 zero-gradient updates out of 300 per seed. Mean gradient norms are 0.090, 0.119, 0.090, 0.097, and 0.100. Reward-log entropy falls on every seed, from 0.633, 0.638, 0.652, 0.596, and 0.711 on the first update to 0.519, 0.514, 0.449, 0.442, and 0.487 on update 300. A zero gradient is a property of the later best-of-K estimator.

On the original training curves, greedy validation duration is shorter at step 25 than at step 300 for every H2 seed. Step 25 versus step 300, in hours: 21.423 to 21.765, 21.636 to 21.760, 21.761 to 21.911, 21.273 to 21.840, and 21.485 to 21.708. Later updates keep producing gradients. They do not keep shortening the greedy route.

## 2. Audit of the original estimator

`reinforce_loss` averages `−A_i log π(τ_i)` over finite trajectories. `A_i` is built as a Python float and then detached. `log π(τ)` is the sum of the route-head and customer-head log-probabilities recorded by `decode`. Feasibility masks remain masks: temperature scales only finite sampling logits, and greedy argmax uses the unscaled logits. `CLOSE` is a legal action in that same factorization. The reward for a feasible route is `−Z_full / (n T_max)`, so ascending reward descends duration. Leave-one-out replaces rollout `i` with the mean of the other rollouts. A single rollout falls back to a zero advantage. Incomplete trajectories with a non-finite log-probability are dropped from the average. There is no reward standardization in the default configuration (`scale_advantages` mode `none`).

The tests in `tests/test_bc_rl.py`, class `ReinforceGradientTests`, check the pieces that can be checked exactly:

- One binary action, costs 1 and 3. The probability-weighted REINFORCE loss matches `−∇ E[R]` within `1e-6`.
- Subtracting 0.7 from every reward leaves that gradient unchanged.
- Central differences of the expected duration, step `1e-6`, match autograd within `1e-8`.
- Two independent binary actions and four enumerated outcomes. Marginal costs are scaled so that `−Δ / (n T_max)` recovers the normalized return. `return_to_go_loss`, weighted by the outcome probability, matches `−∇ E[sum of returns]` on both heads within `1e-6`.
- On random n=6 decodes, the sum of recorded duration changes equals the sum of route-plan durations. When the decode is feasible, that sum equals the exact nonlinear objective.

Checkpoint loading still reads `state_dict` plus the saved config. The correction runs start from the behavior-cloning checkpoint trained on the same 500 instances, with Adam at `2e-5` and gradient clipping at 1. Greedy and sampling share `decode`; sampling is the only path that divides finite logits by temperature, and temperature stayed at 1.

## 3. Why a shorter route does not become the greedy route

The wide limited-discrepancy decoder, already recorded in `results/raw/algorithmic_superiority_v1/guided_diagnostic/`, finds coverable development routes 1.161 h shorter than behavior-cloning greedy (−5.08%). The same decoder on the REINFORCE checkpoint is 0.139 h longer than on behavior cloning. Eight samples from the policy do not find those routes: their best is longer than greedy.

Of 1,016 improving wide deviations, 738 are the second-best customer and 278 are the second-best route. They occur at every decision index: 114, 167, 174, 173, 103, 56, 111, and 118 at decisions 0 through 7. The largest savings are not confined to the last action. Terminal feedback therefore assigns one advantage to a sequence whose useful deviation may have been the second decision.

Log-probabilities of those improving actions, joined in `results/raw/reinforce_diagnosis_v1/better_action_logprobs.csv`:

| Head | Policy | n | Chosen log-probability | Improving alternative | Geometric-mean probability of the alternative |
| --- | --- | ---: | ---: | ---: | ---: |
| Customer | BC | 370 | −1.086 | −1.519 | 0.219 |
| Customer | RL | 368 | −1.072 | −1.554 | 0.211 |
| Route | BC | 132 | −0.382 | −1.443 | 0.236 |
| Route | RL | 146 | −0.256 | −1.875 | 0.153 |

REINFORCE raises the greedy route action (geometric-mean probability 0.682 to 0.774) and lowers the improving second route (0.236 to 0.153). The customer gap moves only from 0.43 nats to 0.48 nats. The route gap moves from 1.06 nats to 1.62 nats. Among improving route actions under REINFORCE, 32.9% have log-probability below −2. The route head is where the learned ranking moves away from the decision that shortens the route.

One coverable example, behavior-cloning seed 46, development instance 12, both endpoints feasible under exact nonlinear charging:

| Route | Duration | Travel | Charging | Customer sequence |
| --- | ---: | ---: | ---: | --- |
| Greedy | 27.230 h | 20.723 h | 1.508 h | `10-4-8-7 \| 6-2-3-5-1 \| 9` |
| Second-best route at decision 1, then greedy | 17.724 h | 12.005 h | 0.719 h | `10-4-2-8-6 \| 3-5-7-9-1` |
| Best of the stochastic samples | 25.210 h |  |  |  |

Service is 5 h on both feasible routes. The route-head alternative at decision 1 saves 9.506 h, of which 8.717 h is travel and 0.789 h is nonlinear charging. Sampling recovers about 2.0 h and misses that split. Customer-head alternatives later in the same trace save 0.08–0.11 h.

One instrumented batch on the first two coverable development instances, H2 seed 42 behavior-cloning weights, four samples, one Adam step at `2e-5`:

| Estimator | Loss | Route-head gradient | Customer-head gradient | Other parameters | Parameter L2 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Leave-one-out | −0.0340 | 0.0133 | 0.00136 | 0.0400 | 0.0139 |
| Return-to-go | −0.0460 | 0.0370 | 0.0121 | 0.1485 | 0.0140 |

The customer head receives the smallest leave-one-out gradient on this batch. Return-to-go increases it by about nine times. That is one batch, not a training result. The full training logs record the total gradient norm and the entropy. They do not record a separate head norm on every step. Cumulative movement from the behavior-cloning checkpoint to the selected REINFORCE checkpoint is an L2 distance of 0.24 to 0.69, depending on the seed (`confirmation_summary.json`). Parameters do move. The greedy argmax on the held-out instances mostly does not.

The same behavior-cloning decode used for the telescope check is feasible at 20.68195604060144 h. The sum of marginal hours, the sum of plan durations, and the objective differ by 0.

## 4. The correction that was trained

`results/raw/reinforce_diagnosis_v1/frozen_protocol.json` was written before the updates. One configuration:

- H2 Route Transformer.
- Behavior cloning for 300 steps at `5e-4`, then REINFORCE for 150 steps at `2e-5`, from that behavior-cloning checkpoint.
- Five seeds, 42–46. Each seed has a leave-one-out run and a return-to-go run.
- Batch 4, 4 rollouts, entropy coefficient 0, KL coefficient 0, gradient clip 1, normalized reward.
- Training instances: generator v2, n=10, scan from seed 48100000. An instance enters training when every customer has a feasible dedicated route and the basic randomized-greedy teacher is feasible. The scan stopped at 680 seeds with 500 accepted instances. All 180 exclusions are physically uncoverable. Every coverable scanned instance had a feasible teacher. The ledger is `instance_ledger.csv`.
- Validation: seeds 49100000–49100039, 40 instances, no teacher filter, 26 physically coverable. Confirmation seeds 49210000–49210039 were named in the protocol and were not used for training or selection.
- Selection score: feasibility first, then the average of penalized greedy duration and the penalized best of greedy plus three samples. Sample seeds for that construction start at 981000.
- Claim rule, fixed in the protocol: return-to-go is promising only if its selected greedy duration is at least 1% below the equally trained behavior-cloning greedy duration on at least 4 of 5 seeds and the paired Student-t 95% interval lies entirely below zero.

`return_to_go_loss` charges a feasible decision `−ΔΦ_t / (n T_max)`, where `Φ` is the sum of current exact route durations after `frvcpy` solves the fixed customer sequences. `CLOSE` has `ΔΦ = 0`. The sum of `ΔΦ` equals the sum of plan durations on every decode, and equals `Z_full` when the trajectory is feasible. The return-to-go is the sum of later rewards, so it equals the normalized feasible reward at the first decision and the remaining duration change at later decisions. The baseline at index `t` is the mean return of the other rollouts that also have a decision `t`. It is a float, so it is detached. The loss averages over trajectories. Dividing by the number of decisions would have changed the estimator; the unit test rejects that version. An infeasible or length-mismatched trajectory puts the existing terminal penalty on its last decision and zero reward before that, so every earlier decision still receives the terminal return. Empty batches return a zero tensor that requires grad.

Across the new runs the gradient is nonzero on all 750 leave-one-out updates and all 750 return-to-go updates. Mean gradient norms are 0.098 and 0.102. Entropy starts at 0.891 for both, because they share the behavior-cloning initialization, and ends at 0.665 for leave-one-out and 0.678 for return-to-go. The policy sharpens throughout. Solver executions logged in the RL reward files are 600,470 and 598,776. Useful updates per 1,000 solver executions are 1.249 and 1.253. Wall time summed from the training logs is 5,209 s. The leave-one-out runs include behavior cloning (about 660–720 s each). The return-to-go runs start from the saved behavior-cloning weights (about 360 s each).

Greedy validation duration is not monotone. Several seeds are shortest at step 50 and longer at step 150. Replacing the composite selection rule by the best logged greedy step changes the paired return-to-go difference by less than 0.001 h. Selection is not hiding a 1% greedy gain.

## 5. Validation and confirmation

Paired differences use five seeds, the Student-t critical value 2.7764 on 4 degrees of freedom, and exact two-sided sign and Wilcoxon tests. A negative difference is a shorter duration. The table is `results/tables/final/reinforce_diagnosis_paired.csv`. The figure is `results/final/figures/fig_reinforce_diagnosis.png`.

Validation, all 40 instances, selected checkpoints. Penalized duration assigns 100 h to each infeasible route. Feasibility stays 0.65 for every logged evaluation, matching the 26 coverable instances.

| Comparison | Mean | 95% interval | Seeds shorter | Relative | t p | Sign p | Wilcoxon p |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| LOO − BC greedy | −0.0168 h | [−0.1525, +0.1188] | 4/5 | −0.033% | 0.748 | 0.375 | 0.625 |
| RTG − BC greedy | −0.0213 h | [−0.1583, +0.1158] | 4/5 | −0.041% | 0.689 | 0.375 | 0.625 |
| RTG − LOO greedy | −0.0044 h | [−0.0204, +0.0116] | 1/5 | −0.009% | 0.484 | 1.000 | 1.000 |
| RTG − BC, feasible mean | −0.0327 h | [−0.2436, +0.1782] | 4/5 | −0.130% | 0.689 | 0.375 | 0.625 |
| RTG − BC, greedy+3 | −0.0966 h | [−0.2271, +0.0338] | 4/5 | −0.190% | 0.109 | 0.375 | 0.125 |

Seed-wise greedy deltas, return-to-go minus behavior cloning, in hours: −0.1229, −0.0562, +0.1677, −0.0554, −0.0395. Seed 44 is longer and is enough to put zero inside the interval. On three seeds the selected greedy duration of return-to-go equals leave-one-out exactly.

Confirmation was scored once, after the checkpoints were frozen. Seeds 49210000–49210039, 40 instances, no teacher filter and no model filter. Twenty-nine are physically coverable. For behavior cloning, leave-one-out, and return-to-go, greedy feasibility is 1 on those 29 and 0 on the other 11. Feasibility and physical coverability coincide. Mean greedy solver use is 54.5 `frvcpy` executions and about 0.05 s. Greedy plus three samples uses about 200 executions and about 0.21 s.

| Comparison | Mean | 95% interval | Seeds shorter | Relative | t p | Sign p | Wilcoxon p |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| RTG − BC greedy, all 40 | +0.0226 h | [−0.0663, +0.1115] | 3/5 | +0.053% | 0.519 | 1.000 | 0.813 |
| RTG − BC greedy, 29 coverable | +0.0312 h | [−0.0914, +0.1538] | 3/5 | +0.148% | 0.519 | 1.000 | 0.813 |
| LOO − BC greedy, 29 coverable | +0.0311 h | [−0.0915, +0.1537] | 3/5 | +0.148% | 0.520 | 1.000 | 0.813 |
| RTG − LOO greedy, all 40 | +0.00002 h | [−0.00004, +0.00009] | 0/5 | ~0 | 0.374 | 1.000 | 1.000 |
| RTG − BC greedy+3, all 40 | −0.1078 h | [−0.3494, +0.1338] | 3/5 | −0.251% | 0.283 | 1.000 | 0.438 |
| RTG − BC sample mean, all 40 | −0.8259 h | [−1.0689, −0.5829] | 5/5 | −1.76% | 0.00070 | 0.0625 | 0.0625 |
| LOO − BC sample mean, all 40 | −0.8499 h | [−1.0259, −0.6738] | 5/5 | −1.81% | 0.00018 | 0.0625 | 0.0625 |
| RTG − LOO sample mean, all 40 | +0.0240 h | [−0.0438, +0.0917] | 2/5 | +0.052% | 0.381 | 1.000 | 0.438 |

On the 29 coverable greedy routes, pooled travel / service / charging is 14.764 / 5.000 / 1.376 h for behavior cloning and 14.772 / 5.000 / 1.398 h for return-to-go. The extra 0.031 h is 0.008 h of travel and 0.023 h of nonlinear charging. Service stays 5 h because every customer is served once.

The sample-mean rows are a diagnostic. Both REINFORCE estimators shorten the average sampled route relative to the equally trained behavior-cloning policy, by about 0.83–0.85 h, with a t interval entirely below zero. The exact sign and Wilcoxon tests on five seeds remain at 0.0625. Return-to-go does not shorten that average relative to leave-one-out. The returned greedy route moves the other way, by +0.023 h. That is the same split the study has already reported: a lower mean over stochastic trajectories is a different object from a shorter greedy route.

## 6. Stop

The development rule fails on the metric it named. Confirmation, which the protocol still required once, fails on that same metric. Shift instances were not decoded. No learning rate, sample count, architecture, temperature, or second credit-assignment run was started. Elite imitation was not added. `paper_draft.md` was not edited.

Raw outputs: `results/raw/reinforce_diagnosis_v1/` (`frozen_protocol.json`, `instance_ledger.csv`, `teacher_cache.pkl`, `gradient_audit.json`, `better_action_logprobs.csv`, `runs/`, `confirmation/`). Tests: `python -m unittest tests.test_bc_rl.ReinforceGradientTests`.
