# Continue Report

Warm-start from `perf_v2b_fc` → short BC on v2c + fleet-CF → POMO →
repair-RL → hybrid LNS (classical ruin floor + neural attempts).
Architecture frozen. No official-instance training. Search labeled.

## Construction basins (pre-LS)

- H2_c3: mean construct 25.0823 h
- H2_cont: mean construct 23.2616 h
- H2_v2b_fc: mean construct 29.0592 h
- H4_c3: mean construct 25.3480 h
- H4_cont: mean construct 24.2817 h
- H4_v2b_fc: mean construct 29.0196 h

## n=10 + strong LS (architecture)

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 19.3647 | 0.634 | 24.8425 |
| H2_cont | 100.0 | 19.6265 | 3.961 | 23.0138 |
| H2_v2b_fc | 100.0 | 19.3695 | 1.150 | 28.4492 |
| H4_c3 | 100.0 | 19.3752 | 0.731 | 25.2040 |
| H4_cont | 100.0 | 19.3967 | 1.355 | 24.1469 |
| H4_v2b_fc | 100.0 | 19.3695 | 1.150 | 28.3144 |

## n=10 + classical ruin (**search**)

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 19.2750 | 0.180 | 24.8425 |
| H2_cont | 100.0 | 19.2603 | 0.052 | 23.0138 |
| H2_v2b_fc | 100.0 | 19.2723 | 0.131 | 28.4492 |
| H4_c3 | 100.0 | 19.2736 | 0.169 | 25.2040 |
| H4_cont | 100.0 | 19.2717 | 0.118 | 24.1469 |
| H4_v2b_fc | 100.0 | 19.2723 | 0.131 | 28.3144 |

## n=10 + hybrid LNS (**search**)

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 19.3036 | 0.335 | 24.8425 |
| H2_cont | 100.0 | 19.4984 | 2.695 | 23.0138 |
| H2_v2b_fc | 100.0 | 19.2675 | 0.107 | 28.4492 |
| H4_c3 | 100.0 | 19.3273 | 0.543 | 25.2040 |
| H4_cont | 100.0 | 19.2769 | 0.184 | 24.1469 |
| H4_v2b_fc | 100.0 | 19.2675 | 0.107 | 28.3144 |

## n=20 + strong LS

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 31.3056 | 11.236 | 39.2235 |
| H2_cont | 100.0 | 30.9838 | 9.022 | 37.2544 |
| H2_v2b_fc | 100.0 | 31.7230 | 12.910 | 44.9095 |
| H4_c3 | 100.0 | 30.5929 | 7.027 | 38.9178 |
| H4_cont | 100.0 | 30.9669 | 8.823 | 39.3950 |
| H4_v2b_fc | 100.0 | 32.0420 | 13.914 | 45.2543 |

## n=20 + classical ruin (**search**)

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 30.7107 | 8.002 | 39.2235 |
| H2_cont | 100.0 | 30.5726 | 6.963 | 37.2544 |
| H2_v2b_fc | 100.0 | 30.6428 | 6.847 | 44.9095 |
| H4_c3 | 100.0 | 30.1793 | 4.674 | 38.9178 |
| H4_cont | 100.0 | 30.3314 | 5.494 | 39.3950 |
| H4_v2b_fc | 100.0 | 30.6335 | 6.500 | 45.2543 |

## n=20 + hybrid LNS (**search**)

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 31.2325 | 11.269 | 39.2235 |
| H2_cont | 100.0 | 31.0631 | 9.101 | 37.2544 |
| H2_v2b_fc | 100.0 | 31.3104 | 11.134 | 44.9095 |
| H4_c3 | 100.0 | 30.9191 | 9.057 | 38.9178 |
| H4_cont | 100.0 | 30.4326 | 6.049 | 39.3950 |
| H4_v2b_fc | 100.0 | 31.2957 | 11.060 | 45.2543 |

## Conclusions

| Goal | Result |
|---|---|
| Better basins vs Cycle-3 | **Yes** — H2_cont 23.26 h vs H2_c3 25.08 h |
| Best n=10 lit gap (search) | **H2_cont + classical ruin ≈ 0.052%** (beats prior 0.13%) |
| H4_cont + ruin | 0.118% (also beats prior v2b_fc 0.13%) |
| n=20 H2 transfer | H2_cont LS 9.0% / ruin 7.0% vs c3 11.2% / 8.0% |
| Best n=20 search | Still **H4_c3 + classical ruin ≈ 4.67%** |
| H4 > H2 matched | **No** on n=10 ruin (H2_cont 0.052 < H4_cont 0.118) |

Hybrid LNS helped v2b_fc (0.107%) but hurt H2_cont vs classical ruin under the lighter finishing LS budget — keep classical ruin as the practical search default. Search gains are not architecture claims.

## Commands

```bash
python results/run_continue.py --stage all
```
