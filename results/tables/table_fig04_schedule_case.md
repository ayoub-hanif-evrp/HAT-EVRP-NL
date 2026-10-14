# Charging schedule case (companion to Figure 4)

- Instance: `tc0c10s2cf1`
- Fixed customer sequences from TRUE-NL H2+LS fixed-route study
- Route index shown: 1 (0-based)
- Customer sequence: `[8, 4, 3, 1]`
- LINEAR-EARLY-RATE schedule source: `tc0c10s2cf1_LINEAR-EARLY-RATE_infeasible_under_true.json`
- TRUE-NL schedule source: `tc0c10s2cf1_TRUE-NL.json`
- SOC traces replayed with **original Montoya nonlinear curves**

| Plan | Route | # charges | True-curve charge h | Energy kWh | Route end h | Tmax h | Duration violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRUE-NL plan (true-physics feasible) | 1 | 2 | 0.4883 | 21.416 | 9.9716 | 10.0 | 0 |
| LINEAR-EARLY-RATE plan replayed on TRUE- | 1 | 2 | 0.6336 | 21.416 | 10.1169 | 10.0 | 1 |

## Full stop lists (node, Δq Wh)

### TRUE-NL

```
[[0, None], [8, None], [4, None], [11, 10321.623005321833], [3, None], [12, 11094.69998373663], [1, None], [0, None]]
```

### LINEAR-EARLY-RATE (replayed on TRUE-NL)

```
[[0, None], [8, None], [4, None], [11, 12721.623005321833], [3, None], [12, 8694.69998373663], [1, None], [0, None]]
```
