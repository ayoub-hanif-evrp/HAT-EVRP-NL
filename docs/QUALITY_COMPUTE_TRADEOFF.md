# Quality versus computation

The mean of four sampled routes estimates the expected duration of one draw from the policy. Evaluating four draws makes that estimate more precise. It does not create a route with the mean duration.

Best-of-four returns one route, the shortest of the four, and it spends the time and the `frvcpy` executions of all four decodes.

Greedy returns one route from one decode.

Numbers below are means over seeds 42–46. Every recorded route is feasible. Gaps use \(Z_{\mathrm{lit}}\) against the BKS column of Table 13 in Froger et al. (2022). That column is not a best-known set verified against later papers. Source: `results/tables/final/deployment_options.csv`, computed from `results/raw/bc_rl_full_v1/eval_stochastic/official_samples.csv`. Figure: `results/final/figures/fig_quality_vs_compute.png`.

## Route Transformer, n=10

| Option | Duration (h) | Time (s) | Executions | Requests | Cache hits | Reference gap |
|---|---:|---:|---:|---:|---:|---:|
| BC greedy | 22.379 | 1.18 | 1,100 | 1,236 | 136 | 23.4% |
| RL greedy | 22.287 | 1.18 | 1,100 | 1,238 | 138 | 22.7% |
| BC best of 4 | 23.085 | 4.90 | 3,875 | 4,370 | 495 | 29.3% |
| RL best of 4 | 22.529 | 4.90 | 3,971 | 4,486 | 515 | 25.3% |

BC greedy is the shortest returned route. RL best-of-four is shorter than BC best-of-four and longer than BC greedy, at about 3.6 times the executions. The expected one-draw durations are 26.232 h (BC) and 25.518 h (RL). Those are not rows in the table above.

## Route Transformer, n=20 and n=40

| Size | BC greedy | RL greedy | BC best of 4 | RL best of 4 | Greedy executions | Best-of-4 executions |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 33.521 h, 3.76 s | 33.586 h, 3.65 s | 38.675 h, 14.76 s | 37.984 h, 14.73 s | 4,121 / 4,200 | 15,409 / 15,647 |
| 40 | 60.897 h, 16.42 s | 59.820 h, 16.37 s | 76.257 h, 61.48 s | 74.902 h, 61.34 s | 16,178 / 16,400 | 61,789 / 62,635 |

Feasibility is 1.0 in every cell. At both sizes the greedy routes are shorter than the best-of-four routes. RL greedy is not shorter than BC greedy by a t-interval that excludes zero. Equal sample counts do not imply equal executions: RL best-of-four uses more executions than BC best-of-four at every size.

## Supplementary classical budgets

`results/tables/classical_matched_v1_summary.csv` records classical greedy-plus-local-search and ruin-and-repair at neural execution counts and at neural wall-clock caps. At the n=10 construction cap, classical search is feasible on 12 of 20 instances. The neural constructor is feasible on 20 of 20. Feasible-only classical means are not ranked against neural means. Under the n=10 time cap, classical ruin-and-repair can match or beat neural construction. At n=20 and above in that file, neural construction remains shorter. Those comparisons are context. Ruin-and-repair is not part of the proposed neural method.
