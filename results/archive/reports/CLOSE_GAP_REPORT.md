# Close-Gap Report

Literature-backed pipeline: Montoya-moment v2c → fleet-CF BC → POMO RL →
neural-LNS search. Architecture frozen. No official-instance training.

## Phase A — v2c distribution gate

- mean depot–customer km: 69.83217799345319
- mean customer–customer km: 54.49094099756472
- frac ≥3 stations: 0.462
- gate passed: True

## Phase B — Construction basins (pre-LS)

- H2_c3: mean construct 24.7618 h
- H2_v2b_fc: mean construct 28.5283 h
- H2_v2c_pomo: mean construct 26.1818 h
- H4_c3: mean construct 25.1603 h
- H4_v2b_fc: mean construct 28.4436 h
- H4_v2c_pomo: mean construct 26.5193 h

## n=10 + strong LS (architecture)

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 19.3647 | 0.634 | 24.8425 |
| H2_v2b_fc | 100.0 | 19.3695 | 1.150 | 28.4492 |
| H2_v2c_pomo | 100.0 | 19.8816 | 5.705 | 26.5689 |
| H4_c3 | 100.0 | 19.3752 | 0.731 | 25.2040 |
| H4_v2b_fc | 100.0 | 19.3695 | 1.150 | 28.3144 |
| H4_v2c_pomo | 100.0 | 19.7945 | 5.164 | 26.9818 |

## n=10 + classical ruin (**search**)

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 19.2750 | 0.180 | 24.8425 |
| H2_v2b_fc | 100.0 | 19.2723 | 0.131 | 28.4492 |
| H2_v2c_pomo | 100.0 | 19.4097 | 1.505 | 26.5689 |
| H4_c3 | 100.0 | 19.2736 | 0.169 | 25.2040 |
| H4_v2b_fc | 100.0 | 19.2723 | 0.131 | 28.3144 |
| H4_v2c_pomo | 100.0 | 19.4123 | 1.522 | 26.9818 |

## n=10 + neural LNS (**search**)

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 19.4822 | 2.287 | 24.8425 |
| H2_v2b_fc | 100.0 | 19.3475 | 0.940 | 28.4492 |
| H2_v2c_pomo | 100.0 | 19.5864 | 2.729 | 26.5689 |
| H4_c3 | 100.0 | 19.4963 | 2.387 | 25.2040 |
| H4_v2b_fc | 100.0 | 19.3475 | 0.940 | 28.3144 |
| H4_v2c_pomo | 100.0 | 19.5779 | 2.676 | 26.9818 |

## n=20 + strong LS

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 31.3056 | 11.236 | 39.2235 |
| H2_v2b_fc | 100.0 | 31.7230 | 12.910 | 44.9095 |
| H2_v2c_pomo | 100.0 | 33.9061 | 24.374 | 46.5889 |
| H4_c3 | 100.0 | 30.5929 | 7.027 | 38.9178 |
| H4_v2b_fc | 100.0 | 32.0420 | 13.914 | 45.2543 |
| H4_v2c_pomo | 100.0 | 34.6954 | 28.277 | 46.6508 |

## n=20 + neural LNS (**search**)

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 32.1828 | 16.920 | 39.2235 |
| H2_v2b_fc | 100.0 | 32.7326 | 18.932 | 44.9095 |
| H2_v2c_pomo | 100.0 | 36.5786 | 38.318 | 46.5889 |
| H4_c3 | 100.0 | 31.7108 | 13.400 | 38.9178 |
| H4_v2b_fc | 100.0 | 33.1580 | 20.737 | 45.2543 |
| H4_v2c_pomo | 100.0 | 36.5234 | 38.316 | 46.6508 |

## Conclusions (measured)

| Goal | Result |
|---|---|
| v2c distribution gate | **Passed** (mean_dc≈69.8, frac3≈0.46) |
| Better construction basins | **Partial win**: H2_v2c_pomo mean construct 26.18 vs H2_v2b_fc 28.53 (c3=24.76) |
| Best n=10 lit gap (search) | **H2/H4_v2b_fc + classical ruin ≈ 0.13%** (still best practical) |
| Neural LNS vs classical ruin | Neural LNS **did not beat** classical ruin on n=10 (v2b_fc 0.94% vs 0.13%) |
| H4 > H2 matched | **No** — H2≈H4 under v2c+POMO and under v2b_fc |
| n=20 transfer | Cycle-3 still strongest; short v2c+POMO pilots regress on n=20 |

Honest takeaway: matching moments + POMO **improves basins**, but the best official quality still comes from **fleet-CF warm-start + classical ruin search**. Neural LNS needs more repair training before it replaces greedy/classical ruin. Do not claim DeepSets wins.

## Claims discipline

- Neural LNS / ruin improvements are **search**, not DeepSets architecture.
- H4 > H2 only if matched v2c+fleet-CF+POMO+same search and layout CI excludes 0.

## Commands

```bash
python results/run_close_gap.py --stage all
```

