# Nonlinear charging features: diagnostic, not a retraining

Reviewed parent: `f42a5b610ef50e5587a48e4432ee851e861741ec`.

This note answers whether the Route Transformer policy has been shown to use nonlinear charging curvature. The diagnostic is `scripts/reporting/diagnose_curve_features.py`. The tables are `results/tables/final/curve_feature_diagnostic.csv` and `curve_feature_diagnostic.json`. No weights were retrained.

## What the network actually receives

`HierarchicalPolicy._node_features` builds four static curve features for each station:

`F(0.25Q) / F(Q)`, `F(0.50Q) / F(Q)`, `F(0.75Q) / F(Q)`, `F(Q) / F(Q)`.

`Q` is 16 kWh. The same node vector also contains a three-way station-type indicator, `fast`, `normal`, or `slow`. Customers and the depot receive zeros in both blocks.

The Montoya breakpoints are 0, 13.6, 15.2, and 16.0 kWh, which are `0`, `0.85Q`, `0.95Q`, and `Q`. The first three sample points therefore lie on the first linear piece. `F(Q) / F(Q)` is identically 1, so the slow tail enters the feature only by making `F(0.75Q) / F(Q)` smaller than 0.75.

On all 60 official instances with 10, 20, or 40 customers, each technology has exactly one normalized vector:

| Technology | `F(0.25Q)/F(Q)` | `F(0.50Q)/F(Q)` | `F(0.75Q)/F(Q)` | `F(Q)/F(Q)` |
| --- | ---: | ---: | ---: | ---: |
| Fast | 0.178777 | 0.357555 | 0.536332 | 1 |
| Normal | 0.180547 | 0.361095 | 0.541642 | 1 |
| Slow | 0.181661 | 0.363322 | 0.544983 | 1 |

Within a technology the four numbers have zero variance. Across technologies the shape vectors differ only in the fourth decimal place. Dividing by full-charge time removes the absolute duration. The physical full-charge times on the same instances are 0.51 h (fast), 1.01 h (normal), and 2.04 h (slow). Cumulative times at 50%, 85%, 95%, and 100% SOC are:

| Technology | 50% | 85% | 95% | 100% | Share of time in the last 15% of energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fast | 0.182 h | 0.310 h | 0.390 h | 0.510 h | 0.392 |
| Normal | 0.365 h | 0.620 h | 0.770 h | 1.010 h | 0.386 |
| Slow | 0.741 h | 1.260 h | 1.540 h | 2.040 h | 0.382 |

The station-type indicator already identifies which of these three curves is present. A grid moved to 50%, 85%, 95%, and 100% and still divided by `F(Q)` would remain a near-constant function of that indicator, because every official and synthetic curve of a given type is the same Montoya curve. Normalizing those times by the 10 h route limit would restore the absolute scale. That scale is also identified by the station-type indicator on this curve family.

## What already varies during decoding

The Route Transformer history is not the static grid. `plan_visit_features` records the realized charge duration and state of charge from the current exact `frvcpy` plan. Those values change with arrival energy, charge amount, and station type. The exact charging solver, not the four static numbers, is what enforces the nonlinear charging time.

## Decision

The hypothesis that the 25/50/75/100 grid is the cause of the deployment gap is not supported as a reason to retrain. The grid does miss the 0.85Q and 0.95Q kinks, and the normalized shape is almost the same for all three technologies. Those facts do not show that the policy is failing because of the sample locations. The type indicator already names the curve, and the history carries the realized charging time.

A matched behavior-cloning and REINFORCE ablation was not run. Replacing the features only at inference would not be evidence that training uses curvature, and a five-seed retraining was not justified by this diagnostic or by the remaining compute budget.

## What is and is not demonstrated

The exact nonlinear solver uses the piecewise-linear charging physics. That was already shown by the fixed-route comparison: at n=40 the true curve stays feasible on 20 of 20 saved routes, while the linear secant and early-rate planners stay feasible on 11 of 20. The saved early-rate case `tc0c10s2cf1` has linear duration 9.971 h and true duration 10.117 h against a 10 h limit.

This diagnostic does not show that the learned policy responds to charging curvature beyond the station-type label. No claim of that kind is made. A curvature-sensitivity test would need curve shapes that vary while the station label stays fixed, with matched training. That experiment was not run, and the official Montoya curves were not altered.
