# Repair-Fix Report

Fixes: open-route neural repair, matched multi-start for all checkpoints,
validation with `cost_lambda=4`, competing_frac ≤ 1, depot/empty route guards.
Architecture frozen. No official training. Search labeled as search.

## Construction protocols n=10 (matched)

| Method | Protocol | Mean best construct h | Mean runtime s | Mean oracle |
|---|---|---:|---:|---:|
| H2_c3 | forced | 22.1007 | 0.48 | 434 |
| H2_c3 | ordinary | 21.7108 | 0.45 | 434 |
| H2_cont | forced | 20.8303 | 0.43 | 440 |
| H2_cont | ordinary | 20.3924 | 0.43 | 440 |
| H2_repair | forced | 20.8224 | 0.43 | 440 |
| H2_repair | ordinary | 20.2841 | 0.45 | 440 |
| H4_c3 | forced | 21.9671 | 0.48 | 423 |
| H4_c3 | ordinary | 21.4858 | 0.45 | 420 |
| H4_cont | forced | 21.1368 | 0.42 | 436 |
| H4_cont | ordinary | 20.6230 | 0.44 | 436 |
| H4_repair | forced | 20.7991 | 0.46 | 436 |
| H4_repair | ordinary | 20.1987 | 0.43 | 435 |

## Matched search n=10 (identical starts/seeds/budgets)

| Method | Mode | Feas% | Mean full h | Mean lit gap% | Mean runtime |
|---|---|---:|---:|---:|---:|
| H2_c3 | classical_ruin | 100.0 | 19.2611 | 0.054 | 0.67 |
| H2_c3 | hybrid_lns | 100.0 | 19.2611 | 0.054 | 0.85 |
| H2_c3 | ls | 100.0 | 19.2729 | 0.150 | 0.65 |
| H2_c3 | neural_lns | 100.0 | 19.2729 | 0.150 | 0.74 |
| H2_cont | classical_ruin | 100.0 | 19.5722 | 3.492 | 0.61 |
| H2_cont | hybrid_lns | 100.0 | 19.5722 | 3.492 | 0.76 |
| H2_cont | ls | 100.0 | 19.7787 | 4.936 | 0.59 |
| H2_cont | neural_lns | 100.0 | 19.7787 | 4.936 | 0.66 |
| H2_repair | classical_ruin | 100.0 | 19.5722 | 3.492 | 0.60 |
| H2_repair | hybrid_lns | 100.0 | 19.5722 | 3.492 | 0.77 |
| H2_repair | ls | 100.0 | 19.7787 | 4.936 | 0.58 |
| H2_repair | neural_lns | 100.0 | 19.7787 | 4.936 | 0.65 |
| H4_c3 | classical_ruin | 100.0 | 19.5654 | 3.423 | 0.64 |
| H4_c3 | hybrid_lns | 100.0 | 19.5654 | 3.423 | 0.82 |
| H4_c3 | ls | 100.0 | 19.5750 | 3.523 | 0.63 |
| H4_c3 | neural_lns | 100.0 | 19.5750 | 3.523 | 0.71 |
| H4_cont | classical_ruin | 100.0 | 19.2717 | 0.118 | 0.65 |
| H4_cont | hybrid_lns | 100.0 | 19.2717 | 0.118 | 0.81 |
| H4_cont | ls | 100.0 | 19.3967 | 1.355 | 0.61 |
| H4_cont | neural_lns | 100.0 | 19.3967 | 1.355 | 0.71 |
| H4_repair | classical_ruin | 100.0 | 19.3315 | 0.821 | 0.71 |
| H4_repair | hybrid_lns | 100.0 | 19.3315 | 0.821 | 0.89 |
| H4_repair | ls | 100.0 | 19.3407 | 0.899 | 0.68 |
| H4_repair | neural_lns | 100.0 | 19.3407 | 0.899 | 0.78 |

### Paired gap deltas n=10 (positive ⇒ classical better)

- H2_c3 hybrid-classical: mean 0.0000 pp (n=20)
- H2_c3 neural-classical: mean 0.0959 pp (n=20)
- H2_cont hybrid-classical: mean 0.0000 pp (n=20)
- H2_cont neural-classical: mean 1.4439 pp (n=20)
- H2_repair hybrid-classical: mean 0.0000 pp (n=20)
- H2_repair neural-classical: mean 1.4439 pp (n=20)
- H4_c3 hybrid-classical: mean 0.0000 pp (n=20)
- H4_c3 neural-classical: mean 0.1003 pp (n=20)
- H4_cont hybrid-classical: mean 0.0000 pp (n=20)
- H4_cont neural-classical: mean 1.2373 pp (n=20)
- H4_repair hybrid-classical: mean 0.0000 pp (n=20)
- H4_repair neural-classical: mean 0.0789 pp (n=20)

## Construction protocols n=20 (matched)

| Method | Protocol | Mean best construct h | Mean runtime s | Mean oracle |
|---|---|---:|---:|---:|
| H2_c3 | forced | 37.3633 | 0.75 | 826 |
| H2_c3 | ordinary | 34.8088 | 0.73 | 832 |
| H2_cont | forced | 35.2423 | 0.66 | 840 |
| H2_cont | ordinary | 33.8927 | 0.69 | 840 |
| H2_repair | forced | 35.1866 | 0.68 | 840 |
| H2_repair | ordinary | 33.7500 | 0.66 | 840 |
| H4_c3 | forced | 36.9106 | 0.71 | 813 |
| H4_c3 | ordinary | 33.9680 | 0.70 | 816 |
| H4_cont | forced | 37.2039 | 0.66 | 830 |
| H4_cont | ordinary | 33.6042 | 0.67 | 830 |
| H4_repair | forced | 37.7204 | 0.72 | 830 |
| H4_repair | ordinary | 34.1362 | 0.67 | 830 |

## Matched search n=20 (identical starts/seeds/budgets)

| Method | Mode | Feas% | Mean full h | Mean lit gap% | Mean runtime |
|---|---|---:|---:|---:|---:|
| H2_c3 | classical_ruin | 100.0 | 30.5107 | 6.396 | 1.30 |
| H2_c3 | hybrid_lns | 100.0 | 30.5107 | 6.396 | 1.61 |
| H2_c3 | ls | 100.0 | 31.4544 | 11.596 | 1.35 |
| H2_c3 | neural_lns | 100.0 | 31.4426 | 11.550 | 1.42 |
| H2_cont | classical_ruin | 100.0 | 31.0382 | 9.751 | 1.19 |
| H2_cont | hybrid_lns | 100.0 | 31.0337 | 9.735 | 1.49 |
| H2_cont | ls | 100.0 | 31.7529 | 13.584 | 1.25 |
| H2_cont | neural_lns | 100.0 | 31.7297 | 13.467 | 1.28 |
| H2_repair | classical_ruin | 100.0 | 31.2198 | 10.636 | 1.17 |
| H2_repair | hybrid_lns | 100.0 | 31.2107 | 10.603 | 1.50 |
| H2_repair | ls | 100.0 | 31.7051 | 13.377 | 1.24 |
| H2_repair | neural_lns | 100.0 | 31.7051 | 13.377 | 1.26 |
| H4_c3 | classical_ruin | 100.0 | 30.7893 | 8.180 | 1.33 |
| H4_c3 | hybrid_lns | 100.0 | 30.7844 | 8.150 | 1.65 |
| H4_c3 | ls | 100.0 | 31.2023 | 10.457 | 1.40 |
| H4_c3 | neural_lns | 100.0 | 31.1987 | 10.444 | 1.44 |
| H4_cont | classical_ruin | 100.0 | 30.9791 | 8.993 | 1.15 |
| H4_cont | hybrid_lns | 100.0 | 30.9790 | 8.992 | 1.48 |
| H4_cont | ls | 100.0 | 32.0241 | 14.924 | 1.22 |
| H4_cont | neural_lns | 100.0 | 31.9749 | 14.635 | 1.26 |
| H4_repair | classical_ruin | 100.0 | 31.4131 | 11.430 | 1.21 |
| H4_repair | hybrid_lns | 100.0 | 31.3786 | 11.228 | 1.55 |
| H4_repair | ls | 100.0 | 32.0478 | 15.175 | 1.28 |
| H4_repair | neural_lns | 100.0 | 32.0055 | 14.957 | 1.33 |

### Paired gap deltas n=20 (positive ⇒ classical better)

- H2_c3 hybrid-classical: mean 0.0000 pp (n=20)
- H2_c3 neural-classical: mean 5.1546 pp (n=20)
- H2_cont hybrid-classical: mean -0.0164 pp (n=20)
- H2_cont neural-classical: mean 3.7156 pp (n=20)
- H2_repair hybrid-classical: mean -0.0329 pp (n=20)
- H2_repair neural-classical: mean 2.7418 pp (n=20)
- H4_c3 hybrid-classical: mean -0.0300 pp (n=20)
- H4_c3 neural-classical: mean 2.2645 pp (n=20)
- H4_cont hybrid-classical: mean -0.0004 pp (n=20)
- H4_cont neural-classical: mean 5.6424 pp (n=20)
- H4_repair hybrid-classical: mean -0.2018 pp (n=20)
- H4_repair neural-classical: mean 3.5269 pp (n=20)

## Conclusions (honest)

| Finding | Evidence |
|---|---|
| Open-route repair bug fixed | Surviving routes stay `closed=False`; unit tests pass |
| Fair old-vs-new decode | Same ordinary + forced schedules for every checkpoint |
| Best matched n=10 search | **H2_c3 + classical ruin ≈ 0.054%** lit gap |
| Prior H2_cont “0.05% win” | Largely an **unequal multi-start artifact**; under matched forced starts H2_cont ruin ≈ 3.5% |
| Neural LNS vs classical | **Does not beat** classical ruin (n10: neural ≈ LS only; hybrid delta ≈ 0 vs classical) |
| Hybrid on n=20 | Tiny hybrid edges on some models (e.g. H4_repair −0.20 pp); not enough to replace classical |
| Practical default | **Keep classical ruin** as the search operator |

Continued / repair-RL checkpoints improve raw construction basins, but under matched post-process the Cycle-3 H2 + classical ruin stack remains strongest on n=10. Neural repair is structurally correct now; it is not yet a quality win.

## Commands

```bash
python results/run_repair_fix.py --stage all
```
