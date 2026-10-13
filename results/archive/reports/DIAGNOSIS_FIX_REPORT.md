# Diagnosis Fix Report

Implements the root-cause analysis fix order: generator rebalance,
construction-basin metrics, fleet-counterfactual imitation, and ruin as
**search-only** post-process.

## 1. Generator v2b (station count + Montoya-like priors)

- Station-count histogram: `{'3': 75, '2': 91, '4': 34}`
- Fraction with ≥3 stations: 0.545
- Tech histogram: `{'fast+slow': 76, 'fast+normal+slow': 18, 'normal+slow': 46, 'slow': 13, 'fast': 19, 'normal': 9, 'fast+normal': 19}`
- Mean depot–customer distance: 65.79 (Montoya n=10 ≈ 69.7; previously collapsed v2 ≈ 57)

Fix: `math.ceil` density mapping + explicit station_count prior; weighted
`fast+slow` / `normal+slow`; offset depot bias. No official coordinates copied.

## 2. Construction basins (pre-LS)

Mean construction objective by method:

- H2_c3: 24.5491 h
- H2_v2b: 28.5196 h
- H2_v2b_fc: 28.3292 h
- H2_v2st: 29.7822 h
- H4_c3: 25.0011 h
- H4_v2b: 28.4551 h
- H4_v2b_fc: 28.3494 h

Artifact: `DIAG_construction_basins_n10.csv`.

## 3. Official n=10 after LS (architecture compare; no ruin)

| Method | Feas% | Mean full h | Mean lit gap% | Mean construct |
|---|---:|---:|---:|---:|
| H2_c3 | 100.0 | 19.3647 | 0.634 | 24.8425 |
| H2_v2b | 100.0 | 19.7273 | 4.558 | 28.4226 |
| H2_v2b_fc | 100.0 | 19.3695 | 1.150 | 28.4492 |
| H4_c3 | 100.0 | 19.3752 | 0.731 | 25.2040 |
| H4_v2b | 100.0 | 19.7471 | 4.719 | 28.1007 |
| H4_v2b_fc | 100.0 | 19.3695 | 1.150 | 28.3144 |

## 4. Official n=10 + ruin (**search**, not architecture)

| Method | Feas% | Mean full h | Mean lit gap% |
|---|---:|---:|---:|
| H2_c3 | 100.0 | 19.2750 | 0.180 |
| H2_v2b | 100.0 | 19.3807 | 0.828 |
| H2_v2b_fc | 100.0 | 19.2723 | 0.131 |
| H4_c3 | 100.0 | 19.2736 | 0.169 |
| H4_v2b | 100.0 | 19.3840 | 0.855 |
| H4_v2b_fc | 100.0 | 19.2723 | 0.131 |

## Conclusions

### What the diagnosis fixed

| Fix | Outcome |
|---|---|
| Station-count bug | 54.5% of n=10 draws now have ≥3 stations (was 0%) |
| Tech priors | `fast+slow` / `normal+slow` dominate (Montoya-like) |
| Construction basins | Logged pre-LS; Cycle-3 still best basin (~24.5 h mean vs ~28–30 for new) |
| Fleet-counterfactual loss | Large practical gain: H2_v2b lit gap 4.56% → **H2_v2b_fc 1.15%** (LS only) |
| Ruin as search | Best practical: **H2/H4_v2b_fc + ruin ≈ 19.272 h / 0.13% lit gap** |

### H4 vs H2 (matched)

Under identical v2b + fleet-CF + λ=4 + strong LS (±ruin):

- LS-only: H2_v2b_fc = H4_v2b_fc (19.3695 h, 1.15% gap)
- +ruin: H2_v2b_fc = H4_v2b_fc (19.2723 h, 0.131% gap)

**Still no DeepSets architectural win.** Fleet-CF improves both H2 and H4 similarly; DeepSets does not add matched benefit. Do not claim H4>H2 from these runs.

### Construction-basin diagnosis stands

Pre-LS mean construction remains worse for new policies than Cycle-3. Post-LS/ruin largely closes the official gap when fleet-CF is used—confirming the analysis that **basin quality + search**, not Transformer redesign, is the bottleneck.

### Labels

- Ruin / `fairls_ruin_search`: **search**, not architecture
- `*_fc` checkpoints: fleet-counterfactual imitation (duration-delta ranking)
- `*_c3.pt`: never overwritten

## Commands

```bash
python results/run_diagnosis_fixes.py --stage all
python evaluate.py compare --size 10 --fair-ls --local-operators strong \
  --cost-lambda 4 --ruin-iterations 6 --ruin-budget 600 --version c3 --methods H2,H4
```

