# Insertion-Aware Neural Repair Report

## Root cause

Open-route neural repair still only **appended** customers via
`oracle.solve((*prefix, customer))`. Classical ruin evaluates all
insertion positions. Position-aware repair keeps the hierarchical
policy for route/customer/new-route choice and uses
`best_feasible_insertion()` (frvcpy at every position) for placement.

## Code changes

- `train.best_feasible_insertion`
- `decode(..., repair_insertion=True|False, oracle=...)`
- `neural_lns` / `hybrid_lns` / `repair_reinforce_step` use insertion mode
- `FixedRouteOracle`: `requests`, `cache_hits`, `solver_executions`
- Matched eval in `results/run_insertion_repair.py` (new CSVs only)

## Regression tests

```bash
python -m unittest tests.test_insertion_repair tests.test_repair_and_validation -v
```

## synth n=20 matched search

| Method | Mode | Feas% | Mean lit gap% | Median | Worst | Mean exec | Mean runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| H2_c3 | classical_ruin | 100.0 | nan | nan | nan | 2825 | 1.06 |
| H2_c3 | hybrid_insert | 100.0 | nan | nan | nan | 3124 | 1.39 |
| H2_c3 | ls | 100.0 | nan | nan | nan | 2498 | 1.07 |
| H2_c3 | neural_append | 100.0 | nan | nan | nan | 2607 | 1.14 |
| H2_c3 | neural_insert | 100.0 | nan | nan | nan | 2905 | 1.38 |
| H2_cont | classical_ruin | 100.0 | nan | nan | nan | 2828 | 1.15 |
| H2_cont | hybrid_insert | 100.0 | nan | nan | nan | 3136 | 1.64 |
| H2_cont | ls | 100.0 | nan | nan | nan | 2492 | 1.21 |
| H2_cont | neural_append | 100.0 | nan | nan | nan | 2588 | 1.25 |
| H2_cont | neural_insert | 100.0 | nan | nan | nan | 2904 | 1.62 |
| H2_repair | classical_ruin | 100.0 | nan | nan | nan | 2833 | 1.19 |
| H2_repair | hybrid_insert | 100.0 | nan | nan | nan | 3139 | 1.69 |
| H2_repair | ls | 100.0 | nan | nan | nan | 2496 | 1.25 |
| H2_repair | neural_append | 100.0 | nan | nan | nan | 2593 | 1.27 |
| H2_repair | neural_insert | 100.0 | nan | nan | nan | 2906 | 1.66 |
| H4_c3 | classical_ruin | 100.0 | nan | nan | nan | 2674 | 1.23 |
| H4_c3 | hybrid_insert | 100.0 | nan | nan | nan | 2980 | 1.73 |
| H4_c3 | ls | 100.0 | nan | nan | nan | 2351 | 1.29 |
| H4_c3 | neural_append | 100.0 | nan | nan | nan | 2460 | 1.35 |
| H4_c3 | neural_insert | 100.0 | nan | nan | nan | 2755 | 1.71 |
| H4_cont | classical_ruin | 100.0 | nan | nan | nan | 2860 | 1.25 |
| H4_cont | hybrid_insert | 100.0 | nan | nan | nan | 3164 | 1.67 |
| H4_cont | ls | 100.0 | nan | nan | nan | 2548 | 1.29 |
| H4_cont | neural_append | 100.0 | nan | nan | nan | 2645 | 1.31 |
| H4_cont | neural_insert | 100.0 | nan | nan | nan | 2968 | 1.62 |
| H4_repair | classical_ruin | 100.0 | nan | nan | nan | 2892 | 1.19 |
| H4_repair | hybrid_insert | 100.0 | nan | nan | nan | 3192 | 1.65 |
| H4_repair | ls | 100.0 | nan | nan | nan | 2568 | 1.26 |
| H4_repair | neural_append | 100.0 | nan | nan | nan | 2668 | 1.30 |
| H4_repair | neural_insert | 100.0 | nan | nan | nan | 2981 | 1.64 |

### Paired gap deltas (synth n=20; + ⇒ classical better)


## official n=10 matched search

| Method | Mode | Feas% | Mean lit gap% | Median | Worst | Mean exec | Mean runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| H2_c3 | classical_ruin | 100.0 | 0.088 | 0.015 | 1.170 | 996 | 0.55 |
| H2_c3 | hybrid_insert | 100.0 | 0.027 | 0.010 | 0.365 | 1140 | 0.77 |
| H2_c3 | ls | 100.0 | 0.097 | 0.015 | 1.353 | 839 | 0.53 |
| H2_c3 | neural_append | 100.0 | 0.097 | 0.015 | 1.353 | 902 | 0.59 |
| H2_c3 | neural_insert | 100.0 | 0.097 | 0.015 | 1.353 | 1046 | 0.74 |
| H2_cont | classical_ruin | 100.0 | 3.531 | 0.017 | 27.034 | 993 | 0.51 |
| H2_cont | hybrid_insert | 100.0 | 3.504 | 0.017 | 26.763 | 1143 | 0.72 |
| H2_cont | ls | 100.0 | 4.936 | 0.287 | 27.034 | 834 | 0.49 |
| H2_cont | neural_append | 100.0 | 4.936 | 0.287 | 27.034 | 896 | 0.55 |
| H2_cont | neural_insert | 100.0 | 4.884 | 0.287 | 27.034 | 1042 | 0.66 |
| H2_repair | classical_ruin | 100.0 | 3.531 | 0.017 | 27.034 | 988 | 0.52 |
| H2_repair | hybrid_insert | 100.0 | 3.504 | 0.017 | 26.763 | 1135 | 0.73 |
| H2_repair | ls | 100.0 | 4.936 | 0.287 | 27.034 | 829 | 0.49 |
| H2_repair | neural_append | 100.0 | 4.936 | 0.287 | 27.034 | 891 | 0.55 |
| H2_repair | neural_insert | 100.0 | 4.884 | 0.287 | 27.034 | 1036 | 0.69 |
| H4_c3 | classical_ruin | 100.0 | 3.514 | 0.017 | 26.518 | 1005 | 0.54 |
| H4_c3 | hybrid_insert | 100.0 | 3.453 | 0.016 | 26.518 | 1150 | 0.77 |
| H4_c3 | ls | 100.0 | 3.657 | 0.018 | 26.865 | 851 | 0.52 |
| H4_c3 | neural_append | 100.0 | 3.657 | 0.018 | 26.865 | 916 | 0.58 |
| H4_c3 | neural_insert | 100.0 | 0.975 | 0.018 | 15.783 | 1056 | 0.73 |
| H4_cont | classical_ruin | 100.0 | 0.438 | 0.018 | 6.247 | 977 | 0.51 |
| H4_cont | hybrid_insert | 100.0 | 0.438 | 0.018 | 6.247 | 1127 | 0.72 |
| H4_cont | ls | 100.0 | 1.675 | 0.045 | 16.080 | 824 | 0.48 |
| H4_cont | neural_append | 100.0 | 1.675 | 0.045 | 16.080 | 886 | 0.56 |
| H4_cont | neural_insert | 100.0 | 1.571 | 0.020 | 16.080 | 1026 | 0.68 |
| H4_repair | classical_ruin | 100.0 | 0.805 | 0.016 | 14.919 | 1019 | 0.53 |
| H4_repair | hybrid_insert | 100.0 | 0.805 | 0.016 | 14.919 | 1158 | 0.75 |
| H4_repair | ls | 100.0 | 1.525 | 0.036 | 14.919 | 858 | 0.51 |
| H4_repair | neural_append | 100.0 | 1.525 | 0.036 | 14.919 | 920 | 0.56 |
| H4_repair | neural_insert | 100.0 | 1.498 | 0.022 | 14.919 | 1064 | 0.71 |

### Paired gap deltas (official n=10; + ⇒ classical better)

- H2_c3 append-classical: mean 0.0091 pp (n=20)
- H2_c3 hybrid-classical: mean -0.0606 pp (n=20)
- H2_c3 insert-append: mean 0.0000 pp (n=20)
- H2_c3 insert-classical: mean 0.0091 pp (n=20)
- H2_cont append-classical: mean 1.4042 pp (n=20)
- H2_cont hybrid-classical: mean -0.0271 pp (n=20)
- H2_cont insert-append: mean -0.0517 pp (n=20)
- H2_cont insert-classical: mean 1.3525 pp (n=20)
- H2_repair append-classical: mean 1.4042 pp (n=20)
- H2_repair hybrid-classical: mean -0.0271 pp (n=20)
- H2_repair insert-append: mean -0.0517 pp (n=20)
- H2_repair insert-classical: mean 1.3525 pp (n=20)
- H4_c3 append-classical: mean 0.1426 pp (n=20)
- H4_c3 hybrid-classical: mean -0.0606 pp (n=20)
- H4_c3 insert-append: mean -2.6820 pp (n=20)
- H4_c3 insert-classical: mean -2.5395 pp (n=20)
- H4_cont append-classical: mean 1.2373 pp (n=20)
- H4_cont hybrid-classical: mean 0.0000 pp (n=20)
- H4_cont insert-append: mean -0.1035 pp (n=20)
- H4_cont insert-classical: mean 1.1337 pp (n=20)
- H4_repair append-classical: mean 0.7203 pp (n=20)
- H4_repair hybrid-classical: mean 0.0000 pp (n=20)
- H4_repair insert-append: mean -0.0269 pp (n=20)
- H4_repair insert-classical: mean 0.6934 pp (n=20)

## official n=20 matched search

| Method | Mode | Feas% | Mean lit gap% | Median | Worst | Mean exec | Mean runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| H2_c3 | classical_ruin | 100.0 | 6.396 | 5.292 | 18.636 | 3062 | 1.15 |
| H2_c3 | hybrid_insert | 100.0 | 6.289 | 5.292 | 18.636 | 3377 | 1.58 |
| H2_c3 | ls | 100.0 | 11.596 | 11.083 | 23.852 | 2723 | 1.22 |
| H2_c3 | neural_append | 100.0 | 11.550 | 10.987 | 23.852 | 2824 | 1.24 |
| H2_c3 | neural_insert | 100.0 | 10.839 | 9.964 | 23.147 | 3141 | 1.54 |
| H2_cont | classical_ruin | 100.0 | 9.751 | 8.221 | 24.465 | 3043 | 1.15 |
| H2_cont | hybrid_insert | 100.0 | 9.330 | 7.542 | 24.465 | 3345 | 1.57 |
| H2_cont | ls | 100.0 | 13.584 | 13.073 | 35.977 | 2711 | 1.21 |
| H2_cont | neural_append | 100.0 | 13.467 | 12.836 | 35.977 | 2803 | 1.23 |
| H2_cont | neural_insert | 100.0 | 11.926 | 12.577 | 29.055 | 3123 | 1.54 |
| H2_repair | classical_ruin | 100.0 | 10.636 | 10.331 | 24.465 | 3040 | 0.99 |
| H2_repair | hybrid_insert | 100.0 | 10.169 | 9.715 | 24.465 | 3342 | 1.35 |
| H2_repair | ls | 100.0 | 13.377 | 12.182 | 35.977 | 2720 | 1.05 |
| H2_repair | neural_append | 100.0 | 13.377 | 12.182 | 35.977 | 2811 | 1.05 |
| H2_repair | neural_insert | 100.0 | 12.032 | 11.104 | 29.055 | 3129 | 1.30 |
| H4_c3 | classical_ruin | 100.0 | 8.180 | 7.462 | 18.636 | 3107 | 1.06 |
| H4_c3 | hybrid_insert | 100.0 | 7.684 | 6.276 | 18.636 | 3435 | 1.46 |
| H4_c3 | ls | 100.0 | 10.457 | 11.924 | 23.852 | 2779 | 1.10 |
| H4_c3 | neural_append | 100.0 | 10.444 | 11.924 | 23.852 | 2876 | 1.13 |
| H4_c3 | neural_insert | 100.0 | 9.302 | 9.788 | 23.147 | 3202 | 1.39 |
| H4_cont | classical_ruin | 100.0 | 8.993 | 6.953 | 19.976 | 2981 | 1.18 |
| H4_cont | hybrid_insert | 100.0 | 8.398 | 6.770 | 18.300 | 3302 | 1.63 |
| H4_cont | ls | 100.0 | 14.924 | 12.232 | 36.000 | 2659 | 1.24 |
| H4_cont | neural_append | 100.0 | 14.635 | 12.232 | 36.000 | 2747 | 1.25 |
| H4_cont | neural_insert | 100.0 | 12.964 | 10.005 | 33.751 | 3060 | 1.57 |
| H4_repair | classical_ruin | 100.0 | 11.430 | 10.970 | 22.807 | 3063 | 1.23 |
| H4_repair | hybrid_insert | 100.0 | 10.877 | 10.970 | 22.658 | 3388 | 1.69 |
| H4_repair | ls | 100.0 | 15.175 | 12.525 | 33.884 | 2735 | 1.30 |
| H4_repair | neural_append | 100.0 | 14.957 | 12.525 | 33.884 | 2827 | 1.30 |
| H4_repair | neural_insert | 100.0 | 13.727 | 12.429 | 32.492 | 3150 | 1.65 |

### Paired gap deltas (official n=20; + ⇒ classical better)

- H2_c3 append-classical: mean 5.1546 pp (n=20)
- H2_c3 hybrid-classical: mean -0.1071 pp (n=20)
- H2_c3 insert-append: mean -0.7117 pp (n=20)
- H2_c3 insert-classical: mean 4.4429 pp (n=20)
- H2_cont append-classical: mean 3.7156 pp (n=20)
- H2_cont hybrid-classical: mean -0.4207 pp (n=20)
- H2_cont insert-append: mean -1.5408 pp (n=20)
- H2_cont insert-classical: mean 2.1749 pp (n=20)
- H2_repair append-classical: mean 2.7418 pp (n=20)
- H2_repair hybrid-classical: mean -0.4667 pp (n=20)
- H2_repair insert-append: mean -1.3456 pp (n=20)
- H2_repair insert-classical: mean 1.3962 pp (n=20)
- H4_c3 append-classical: mean 2.2645 pp (n=20)
- H4_c3 hybrid-classical: mean -0.4952 pp (n=20)
- H4_c3 insert-append: mean -1.1426 pp (n=20)
- H4_c3 insert-classical: mean 1.1219 pp (n=20)
- H4_cont append-classical: mean 5.6424 pp (n=20)
- H4_cont hybrid-classical: mean -0.5946 pp (n=20)
- H4_cont insert-append: mean -1.6710 pp (n=20)
- H4_cont insert-classical: mean 3.9714 pp (n=20)
- H4_repair append-classical: mean 3.5269 pp (n=20)
- H4_repair hybrid-classical: mean -0.5530 pp (n=20)
- H4_repair insert-append: mean -1.2293 pp (n=20)
- H4_repair insert-classical: mean 2.2976 pp (n=20)

## official n=40 matched search

| Method | Mode | Feas% | Mean lit gap% | Median | Worst | Mean exec | Mean runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| H2_c3 | classical_ruin | 100.0 | 46.697 | 45.174 | 74.449 | 6011 | 4.33 |
| H2_c3 | hybrid_insert | 100.0 | 41.724 | 43.037 | 56.864 | 6767 | 5.49 |
| H2_c3 | ls | 100.0 | 60.202 | 61.527 | 79.689 | 5611 | 4.30 |
| H2_c3 | neural_append | 100.0 | 57.267 | 60.683 | 73.794 | 5769 | 4.63 |
| H2_c3 | neural_insert | 100.0 | 53.264 | 52.053 | 75.301 | 6402 | 5.38 |
| H2_cont | classical_ruin | 100.0 | 57.517 | 58.167 | 82.549 | 6116 | 3.40 |
| H2_cont | hybrid_insert | 100.0 | 52.790 | 50.433 | 77.709 | 6867 | 4.39 |
| H2_cont | ls | 100.0 | 63.581 | 61.026 | 91.666 | 5735 | 3.45 |
| H2_cont | neural_append | 100.0 | 60.976 | 59.245 | 91.264 | 5885 | 3.61 |
| H2_cont | neural_insert | 100.0 | 58.028 | 54.570 | 90.384 | 6571 | 4.29 |
| H2_repair | classical_ruin | 100.0 | 54.014 | 55.149 | 77.709 | 6118 | 3.34 |
| H2_repair | hybrid_insert | 100.0 | 49.284 | 49.888 | 77.709 | 6869 | 4.27 |
| H2_repair | ls | 100.0 | 62.089 | 60.343 | 91.666 | 5736 | 3.35 |
| H2_repair | neural_append | 100.0 | 59.670 | 60.343 | 91.264 | 5883 | 3.54 |
| H2_repair | neural_insert | 100.0 | 58.033 | 58.292 | 90.384 | 6572 | 4.23 |
| H4_c3 | classical_ruin | 100.0 | 55.930 | 57.118 | 80.498 | 6002 | 3.63 |
| H4_c3 | hybrid_insert | 100.0 | 50.895 | 54.492 | 73.706 | 6771 | 4.44 |
| H4_c3 | ls | 100.0 | 65.073 | 66.177 | 90.744 | 5606 | 3.66 |
| H4_c3 | neural_append | 100.0 | 62.339 | 62.441 | 89.076 | 5774 | 3.82 |
| H4_c3 | neural_insert | 100.0 | 58.049 | 60.424 | 85.345 | 6450 | 4.29 |
| H4_cont | classical_ruin | 100.0 | 56.989 | 58.623 | 79.584 | 6094 | 3.30 |
| H4_cont | hybrid_insert | 100.0 | 53.006 | 54.249 | 75.214 | 6825 | 4.05 |
| H4_cont | ls | 100.0 | 64.300 | 63.237 | 92.534 | 5704 | 3.26 |
| H4_cont | neural_append | 100.0 | 62.663 | 61.768 | 91.022 | 5844 | 3.44 |
| H4_cont | neural_insert | 100.0 | 60.299 | 59.183 | 89.712 | 6545 | 3.93 |
| H4_repair | classical_ruin | 100.0 | 56.565 | 53.515 | 80.795 | 6053 | 2.76 |
| H4_repair | hybrid_insert | 100.0 | 51.862 | 50.784 | 77.164 | 6806 | 3.46 |
| H4_repair | ls | 100.0 | 67.313 | 67.327 | 90.754 | 5669 | 2.76 |
| H4_repair | neural_append | 100.0 | 64.539 | 59.337 | 90.611 | 5810 | 2.89 |
| H4_repair | neural_insert | 100.0 | 60.588 | 56.388 | 90.114 | 6463 | 3.40 |

### Paired gap deltas (official n=40; + ⇒ classical better)

- H2_c3 append-classical: mean 10.5705 pp (n=20)
- H2_c3 hybrid-classical: mean -4.9730 pp (n=20)
- H2_c3 insert-append: mean -4.0030 pp (n=20)
- H2_c3 insert-classical: mean 6.5675 pp (n=20)
- H2_cont append-classical: mean 3.4583 pp (n=20)
- H2_cont hybrid-classical: mean -4.7274 pp (n=20)
- H2_cont insert-append: mean -2.9479 pp (n=20)
- H2_cont insert-classical: mean 0.5105 pp (n=20)
- H2_repair append-classical: mean 5.6563 pp (n=20)
- H2_repair hybrid-classical: mean -4.7300 pp (n=20)
- H2_repair insert-append: mean -1.6366 pp (n=20)
- H2_repair insert-classical: mean 4.0196 pp (n=20)
- H4_c3 append-classical: mean 6.4085 pp (n=20)
- H4_c3 hybrid-classical: mean -5.0350 pp (n=20)
- H4_c3 insert-append: mean -4.2901 pp (n=20)
- H4_c3 insert-classical: mean 2.1185 pp (n=20)
- H4_cont append-classical: mean 5.6736 pp (n=20)
- H4_cont hybrid-classical: mean -3.9838 pp (n=20)
- H4_cont insert-append: mean -2.3634 pp (n=20)
- H4_cont insert-classical: mean 3.3102 pp (n=20)
- H4_repair append-classical: mean 7.9746 pp (n=20)
- H4_repair hybrid-classical: mean -4.7028 pp (n=20)
- H4_repair insert-append: mean -3.9510 pp (n=20)
- H4_repair insert-classical: mean 4.0236 pp (n=20)

## New insert-aware trained checkpoints (`perf_insert_v1`)

Warm-start from Cycle-3 + BC/POMO/repair-RL with `repair_insertion=True`.

| Method | Mode | n=10 lit gap% | n=20 lit gap% |
|---|---|---:|---:|
| H2_insert | hybrid_insert | 0.016 | 6.906 |
| H2_insert | classical_ruin | 0.076 | 7.005 |
| H4_insert | hybrid_insert | 0.808 | 6.678 |
| H4_insert | classical_ruin | 0.868 | 7.218 |

## Conclusions (measured)

| Finding | Evidence |
|---|---|
| Root cause fixed | Best-position frvcpy insertion; append-only preserved when `repair_insertion=False` |
| Insert > append | n=20/40: neural_insert consistently better than neural_append |
| Hybrid beats classical | n=10 H2_c3: **0.027%** vs 0.088%; n=20: **6.289%** vs 6.396%; n=40: **41.7%** vs 46.7% (−5.0 pp) |
| Best retained config | **H2_c3 + strong LS + classical ruin + position-aware neural hybrid** |
| H4 vs H2 matched | No DeepSets win on best n=10/n=20 stacks (H2_c3 hybrid strongest) |
| Extra compute | Hybrid uses ~10% more oracle executions; still better quality |
| Practical default | Switch search default to **hybrid_insert** (classical floor + position-aware neural) |

Search gains are **search**, not architecture. Feasibility 100% on all reported official runs.

## Commands

```bash
python -m unittest tests.test_insertion_repair tests.test_repair_and_validation -v
python results/run_insertion_repair.py --stage all
```
