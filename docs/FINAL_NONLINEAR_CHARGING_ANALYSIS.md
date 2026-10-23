# Final nonlinear charging analysis

The locked evaluation is commit `7671f6411a37666212656121d02a7c3af41ce1f7`. This analysis is the following commit. No BC or RL training was run.

Charging time on station technology \(s\) is \(T_{\mathrm{charge}}=F_s(q_{\mathrm{out}})-F_s(q_{\mathrm{in}})\). \(F_s\) is the original piecewise-linear cumulative curve. The full objective is travel + service + that charging time. The literature objective subtracts customer service and is used only for best-known-solution gaps.

## Curves

Instance `tc2c20s4cf0` contains slow, normal, and fast curves. Each curve has breakpoints at 0, 13.6, 15.2, and 16.0 kWh. The last 0.8 kWh takes 0.50 h on slow, 0.24 h on normal, and 0.12 h on fast, which is slower per kWh than the first segment. Figure: `results/final/figures/fig_charging_curves.png`. Data: `results/tables/final/fig_charging_curves.csv`.

Regression tests in `tests/test_charging_breakpoints.py` cover a partial charge inside one segment, a charge across one breakpoint, a charge across two breakpoints, and a late state-of-charge increment that takes longer than the same energy on the first segment.

## Two different experiments

Both were recounted from the instance files into `results/tables/final/nonlinear_instance_audit.csv`. The summary file agrees. Both use the Cycle-3 Route Transformer checkpoint `H2_n10_seed42_c3.pt`, not the final BC/RL weights.

Fixed-route comparison holds the customer sequences constant and changes the charging planner. Feasible under the original curves:

| Customers | Exact nonlinear | Linear secant | Linear early rate |
|---:|---:|---:|---:|
| 10 | 20/20 | 16/20 | 16/20 |
| 20 | 20/20 | 12/20 | 12/20 |
| 40 | 20/20 | 11/20 | 11/20 |

Planner feasibility is higher than true-physics feasibility for some linear rows. At n=10 the early-rate planner accepts 20/20 and the true check accepts 16/20: 4 duration violations, 0 state-of-charge failures, 0 missing plans. The secant planner accepts 17/20: 3 missing plans and 1 duration violation. At n=40 the early-rate planner accepts 20/20 and the true check accepts 11/20, with 9 duration violations. State-of-charge failures are 0 in every recounted cell. A later exact reoptimization of a rejected linear plan is feasible on 20/20. That repair is a separate method.

End-to-end comparison lets the charging model change the constructed customer sequences. True-physics feasibility:

| Customers | Exact nonlinear | Linear secant | Linear early rate |
|---:|---:|---:|---:|
| 10 | 20/20 | 19/20 | 15/20 |
| 20 | 20/20 | 19/20 | 17/20 |
| 40 | 20/20 | 11/20 | 9/20 |

At n=40 the secant planner itself fails on `tc1c40s5ct1`. Eight further secant plans violate duration under the original curves. Feasible-only means are not used to rank these rows.

## A saved duration violation

`results/raw/nonlinear_charging/schedules/tc0c10s2cf1_LINEAR-EARLY-RATE_infeasible_under_true.json` stores an early-rate plan. Route 1 of that plan is feasible under the early-rate curve at 9.971 h. The same stops and charge amounts take 10.117 h on the original curve. The limit is 10 h. Charging time is 0.488 h versus 0.634 h. The duration gap equals the charging gap. Figure: `results/final/figures/fig_linear_plan_infeasible.png`, with the duration axis restricted to 9.90–10.20 h.

A feasible exact schedule for the same instance is `tc0c10s2cf1_TRUE-NL.json`. One route charges twice at fast stations: 3.28 kWh to 13.60 kWh in 0.235 h, then 2.51 kWh to 13.60 kWh in 0.253 h. Figure: `results/final/figures/fig_soc_schedule.png`.

## Final BC and RL checkpoints

`scripts/experiments/eval_route_decomposition.py` decoded the saved H2 and H4 behavior-cloning and REINFORCE checkpoints, seeds 42–46, official n=10 and n=20, greedy and four samples, with the locked sample seeds. It did not train. All 4,000 regenerated full durations match `official_samples.csv`. Every route is feasible, and travel + service + nonlinear charging matches the simulated duration. Service is 5 h at n=10 and 10 h at n=20, so the paired service difference is 0.

Mean of four samples, RL minus BC, hours. Intervals use the five training seeds.

| Model | n | Travel | Charging | Full | Sequences changed |
|---|---:|---:|---:|---:|---:|
| H2 | 10 | −0.681 [−0.893, −0.469] | −0.033 [−0.066, −0.0001] | −0.714 [−0.944, −0.484] | 43% |
| H2 | 20 | −0.520 [−0.871, −0.170] | −0.045 [−0.109, +0.018] | −0.566 [−0.965, −0.166] | 59% |
| H4 | 10 | −1.361 [−1.902, −0.819] | −0.072 [−0.094, −0.050] | −1.432 [−1.983, −0.882] | 77% |
| H4 | 20 | −0.742 [−1.277, −0.208] | +0.005 [−0.091, +0.100] | −0.738 [−1.328, −0.148] | 84% |

The full-duration column matches the locked official sample table. The sampled improvement is shorter travel. Charging time falls slightly at n=10 and is not established at n=20. At n=20, H4 charging time is slightly higher. Charging visits fall at n=10 and are inconclusive at n=20.

Greedy H4 at n=10 moves charging time by +0.141 h, interval [+0.013, +0.270], on 0 of 5 seeds. Its full-duration interval includes zero.

REINFORCE changes customer sequences. `frvcpy` then chooses stations and charge amounts for those sequences. The network does not output a charge.

## What was not run

A five-seed comparison of exact, secant, and early-rate planning on the frozen final checkpoints was not run. The fixed-route and end-to-end counts above remain the Cycle-3 evidence. The final-checkpoint evidence is the true-curve decomposition, not a new surrogate study.
