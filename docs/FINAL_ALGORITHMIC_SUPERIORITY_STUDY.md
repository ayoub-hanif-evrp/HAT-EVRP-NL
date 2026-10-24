# Final algorithmic superiority study

Starting commit `5fa3e382f6cb408fd35edf10714be693cb1bad0c`. The original checkpoints and the official Montoya evaluation were not modified. No confirmation or shift split was decoded. The manuscript was not changed, because neither trained policy met the predeclared rule for a new claim.

## Answer

REINFORCE does not outperform behavior cloning on the returned route. A neural-guided decoder can find shorter routes than greedy behavior cloning, but those routes are a property of the decoder. Under the same decoder and the same solver budget, the REINFORCE checkpoint is not better than behavior cloning. Training the behavior-cloning policy to imitate those better routes, with or without a REINFORCE term, does not produce a greedy policy that is 1% shorter.

1. **Returned route.** On 29 physically coverable development instances, the selected REINFORCE-plus-imitation greedy route differs from behavior-cloning greedy by −0.062 h, interval [−0.280, +0.156], on 3 of 5 seeds (−0.27%). The interval includes zero.
2. **Value added by REINFORCE.** Against the imitation-only control, the difference is −0.0002 h, interval [−0.0009, +0.0004], shorter on 1 of 5 seeds. Under the wide guided decoder, REINFORCE minus behavior cloning is +0.139 h, interval [−0.116, +0.394], shorter on 1 of 5 seeds.
3. **Same inference budget.** The trained comparison above uses one greedy decode for every method. It is not a 1% improvement. The guided decoder’s reduction uses more charging-solver executions than greedy (179 versus 55 on the behavior-cloning rows) and is reported separately.
4. **Replication.** All five development seeds were run. The RL candidate did not pass the development rule, so the study did not add seeds and did not decode the locked confirmation set.
5. **Other distributions.** Montoya-like, slow-charger, and geographic instances were generated and hashed. They were not evaluated.
6. **Feasibility.** Every retained route was checked with the exact nonlinear charging model. On this development set the greedy and guided decoders are feasible on exactly the instances in which every customer has a feasible dedicated route: 29 of 40. The other 11 stay infeasible for behavior cloning, REINFORCE, and both guided budgets.

## What was predeclared

`results/raw/algorithmic_superiority_v1/splits/protocol.json` was written before the instances. `guided_diagnostic/frozen_protocol.json` was written before any development decode. The practical bar was a 1% shorter returned route, at least 4 of 5 seeds, and a two-sided Student-t 95% interval entirely below zero. `self_imitation/frozen_protocol.json` was written before the first training update. The RL claim had to beat the imitation-only control on that same bar, not only the original behavior-cloning checkpoint.

Development instances are synthetic n=10, generator v2, seeds 41100000–41100039. Training instances are seeds 46100000–46100039. Confirmation seeds start at 42000000. None of these bands were used in the earlier unfiltered study.

## Phase 1 — what the frozen policies are doing

Checkpoints are H2 `bc_best.pt` and `rl_best.pt` from `results/raw/bc_rl_full_v1`, seeds 42–46. Raw rows: `results/raw/algorithmic_superiority_v1/guided_diagnostic/evaluations.csv` (400 rows), `divergences.csv` (200), `deviations.csv`.

Twenty-nine of 40 instances are physically coverable. Behavior-cloning greedy is feasible on those 29 and infeasible on the other 11, for every seed. Mean greedy entropy is 0.620 for behavior cloning and 0.584 for REINFORCE. Eight independent samples from a policy produced eight distinct customer sequences on average (duplicate fraction 0). The best of those eight is longer than that policy’s greedy route: +0.669 h for behavior cloning, interval [+0.205, +1.133], on 0 of 5 seeds. Sampling is exploring other sequences. It is not finding better ones.

The greedy customer sequences of the two checkpoints match on 146 of 200 seed-instance pairs. REINFORCE greedy minus behavior-cloning greedy on the coverable subset is +0.111 h, interval [−0.315, +0.536], shorter on 3 of 5 seeds.

One positive cost coefficient, 4, was compared with the zero control. It lengthens the behavior-cloning greedy route by +1.422 h on the coverable subset, interval [+1.168, +1.676], on 0 of 5 seeds. The coefficient was not searched again.

## Phase 2 — neural-guided construction

The decoder is limited discrepancy, not a second training objective. Along the greedy trace it tries the policy’s second-best legal customer action, and in the wider budget also the second-best legal route action, then finishes with ordinary greedy decoding. There is no local search, insertion, or ruin-and-repair. The returned route is the shortest feasible completion inside the budget.

| Comparison on 29 coverable instances | Mean difference | 95% interval | Seeds shorter | Relative |
| --- | ---: | --- | ---: | ---: |
| Narrow BC guided − BC greedy | −0.822 h | [−1.078, −0.565] | 5/5 | −3.60% |
| Wide BC guided − BC greedy | −1.161 h | [−1.393, −0.929] | 5/5 | −5.08% |
| Best of BC and RL wide − BC greedy | −1.233 h | [−1.484, −0.983] | 5/5 | −5.40% |
| Wide RL guided − wide BC guided | +0.139 h | [−0.116, +0.394] | 1/5 | +0.64% |

The five wide behavior-cloning seed differences are −0.976, −1.044, −1.150, −1.170, and −1.464 h. The same comparison on all 40 instances, with infeasible routes penalized at 100 h, is −0.842 h, interval [−1.010, −0.673], −1.91%. The sign test and Wilcoxon test on five seeds remain p = 0.0625.

Mean charging-solver executions on the behavior-cloning rows are 55 for greedy, 141 for the narrow budget, and 179 for the wide budget. Across the 145 coverable behavior-cloning rows, the wide route is shorter mainly in travel (−1.025 h) and only modestly in nonlinear charging time (−0.136 h). The network still does not choose stations or charge amounts. `frvcpy` does.

Of 5,301 wide-budget deviations, 1,016 finish shorter than greedy (19.2%). They occur at every inspected decision, from the first through the eighth, and more often from the second-best customer (738) than the second-best route (278). Behavior cloning and REINFORCE contribute 502 and 514 of those improvements. The headroom is in the decoder, and both policies can use it.

This passed the predeclared gate for attempting training. It did not pass a claim that REINFORCE preferences are better under a matched decoder.

## Phase 3 — self-imitation and REINFORCE

Both modes start from the saved behavior-cloning checkpoint. For 80 updates, batch size 2, each visit runs the narrow guided decoder. A completed route is an elite only when it beats the frozen behavior-cloning greedy objective on that training instance by more than 0.0001 h. The imitation term is `0.1` times the negative log-probability of that elite action sequence. Labels and advantages are detached. `rl_imitation` also applies leave-one-out REINFORCE to two on-policy samples. `imitation` runs the same two samples so the solver work matches, and does not put them in the loss. Learning rate `2e-5`, gradient clip 1, entropy coefficient 0. Selection keeps the lowest penalized greedy mean on all 40 development instances.

Each run found 75–85 elites. Solver executions were about 24,000 per run. Wall time was 54–79 s per run, 687 s for all ten runs.

Selected coverable greedy means:

| Seed | BC greedy | Imitation | REINFORCE + imitation |
| ---: | ---: | ---: | ---: |
| 42 | 22.652 h | 22.652 h | 22.652 h |
| 43 | 22.744 h | 22.670 h | 22.670 h |
| 44 | 22.769 h | 22.945 h | 22.945 h |
| 45 | 22.965 h | 22.657 h | 22.657 h |
| 46 | 23.042 h | 22.938 h | 22.937 h |

Imitation minus behavior cloning is −0.062 h, interval [−0.280, +0.156], 3 of 5 seeds, −0.27%. REINFORCE plus imitation minus imitation is −0.0002 h, interval [−0.0009, +0.0004], 1 of 5 seeds. The RL rule fails. Feasibility stays 0.725 for every selected checkpoint, the same coverability rate.

## What was not run

The exact marginal-duration reward was not trained. Elite imitation already tells the policy which actions produced the shorter routes, and the greedy development route still did not move by 1%. A second reward would not identify a separate cause. Confirmation, the shift splits, n=20, n=40, and the official Montoya set were not decoded for these new weights. No further learning-rate, temperature, or coefficient run was started.

## Ledger

| Experiment | Status | Result |
| --- | --- | --- |
| Guided-decoder diagnostic on frozen H2 checkpoints | PASS | Wide guided behavior cloning is 5.08% shorter than its own greedy route on coverable development instances. Independent samples are not. |
| Same decoder, REINFORCE versus behavior cloning | NEGATIVE | +0.139 h. REINFORCE preferences are not better at the same budget. |
| `cost_lambda = 4` versus 0 | NEGATIVE | The positive coefficient lengthens the greedy route. |
| Imitation and REINFORCE-plus-imitation | NEGATIVE | Ten runs finished. Neither beats behavior-cloning greedy by the 1% rule. REINFORCE adds nothing beyond imitation. |
| Marginal reward-to-go | NOT RUN | Not identified as the missing mechanism after imitation failed. |
| Confirmation and shift evaluation | NOT RUN | Stopped because the RL candidate failed on development. |

## Artifacts

- `results/raw/algorithmic_superiority_v1/`
- `results/tables/final/guided_decoder_paired.csv`
- `results/tables/final/self_imitation_paired.csv`
- `results/final/figures/fig_guided_decoder_development.png`
- `scripts/experiments/run_guided_diagnostic.py`
- `scripts/experiments/run_self_imitation.py`
- `scripts/reporting/summarize_guided_diagnostic.py`
- `scripts/reporting/plot_guided_decoder.py`

## Limitations

Five seeds cannot produce a two-sided sign-test p-value below 0.0625, so the guided-decoder t-intervals are not a 5% sign-test result. The 5% decoder reduction spends about three times the greedy solver executions. The recorded search runtime is the combined narrow-plus-wide search, not a separate timer for each budget. Training used one coefficient, one learning rate, and 80 updates, as frozen before the run. Elites were available, and they did not transfer into a better greedy route on the development instances. The 11 uncoverable instances are a generator fact. They are included in the penalized means and excluded only from the coverable quality table, by the dedicated-route test, not by which policy failed.
