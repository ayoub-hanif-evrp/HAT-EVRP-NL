# Nonlinear vs linear charging sensitivity (true-physics evaluation)

- Objective convention: All reported full/charging hours use TRUE Montoya nonlinear physics
- Units: hours / kWh; runtime seconds; oracle = frvcpy FixedRouteOracle executions
- Protocol: Fixed-route: matched H2+LS sequences; E2E: same H2 weights, different planning curves
- Source raw: results/raw/nonlinear_charging/
- Seeds: 42
- Checkpoint: H2_n10_seed42_c3.pt
- Coverage: 20 official instances each for n=10,20,40

| Benchmark size | Charging model | Method | Instances | Feasible under true physics | Duration violations | Mean true charging h (feas only) | Mean realized full h (feas only) | Charged energy kWh (feas only) | Charger visits (feas only) | Mean runtime s | Mean oracle executions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | LINEAR-EARLY-RATE | plan_LINEAR-EARLY-RATE | 20 | 16/20 | 4 | 1.8525 | 20.3610 | 26.568 | 3.44 | 0.0009 | 2.75 |
| 10 | LINEAR-EARLY-RATE | repair_TRUE-NL_after_LINEAR-EARLY-RATE | 20 | 20/20 | 0 | 1.5196 | 20.4031 | 28.250 | 3.60 | 0.0008 | 2.75 |
| 10 | LINEAR-SECANT | plan_LINEAR-SECANT | 20 | 16/20 | 1 | 1.8525 | 20.3610 | 26.568 | 3.44 | 0.0010 | 2.75 |
| 10 | LINEAR-SECANT | repair_TRUE-NL_after_LINEAR-SECANT | 20 | 20/20 | 0 | 1.5196 | 20.4031 | 28.250 | 3.60 | 0.0009 | 2.75 |
| 10 | TRUE-NL | plan_TRUE-NL | 20 | 20/20 | 0 | 1.5196 | 20.4031 | 28.250 | 3.60 | n/a | 2.75 |
| 20 | LINEAR-EARLY-RATE | plan_LINEAR-EARLY-RATE | 20 | 12/20 | 4 | 1.5866 | 30.1009 | 31.663 | 4.58 | 0.0013 | 4.05 |
| 20 | LINEAR-EARLY-RATE | repair_TRUE-NL_after_LINEAR-EARLY-RATE | 20 | 20/20 | 0 | 1.6065 | 31.5896 | 37.831 | 5.25 | 0.0014 | 4.05 |
| 20 | LINEAR-SECANT | plan_LINEAR-SECANT | 20 | 12/20 | 0 | 1.5835 | 30.1013 | 31.681 | 4.58 | 0.0014 | 4.05 |
| 20 | LINEAR-SECANT | repair_TRUE-NL_after_LINEAR-SECANT | 20 | 20/20 | 0 | 1.6065 | 31.5896 | 37.831 | 5.25 | 0.0015 | 4.05 |
| 20 | TRUE-NL | plan_TRUE-NL | 20 | 20/20 | 0 | 1.6065 | 31.5896 | 37.831 | 5.25 | n/a | 4.05 |
| 40 | LINEAR-EARLY-RATE | plan_LINEAR-EARLY-RATE | 20 | 11/20 | 9 | 2.5150 | 49.8138 | 34.430 | 7.82 | 0.0056 | 7.05 |
| 40 | LINEAR-EARLY-RATE | repair_TRUE-NL_after_LINEAR-EARLY-RATE | 20 | 20/20 | 0 | 3.0265 | 57.1133 | 59.256 | 9.80 | 0.0059 | 7.05 |
| 40 | LINEAR-SECANT | plan_LINEAR-SECANT | 20 | 11/20 | 1 | 2.5150 | 49.8138 | 34.430 | 7.82 | 0.0057 | 7.05 |
| 40 | LINEAR-SECANT | repair_TRUE-NL_after_LINEAR-SECANT | 20 | 20/20 | 0 | 3.0265 | 57.1133 | 59.256 | 9.80 | 0.0077 | 7.05 |
| 40 | TRUE-NL | plan_TRUE-NL | 20 | 20/20 | 0 | 3.0265 | 57.1133 | 59.256 | 9.80 | n/a | 7.05 |
| 10 | LINEAR-EARLY-RATE | e2e | 20 | 15/20 | 5 | 1.5967 | 21.4884 | 28.389 | 3.60 | 0.0658 | n/a |
| 10 | LINEAR-SECANT | e2e | 20 | 19/20 | 1 | 2.2210 | 23.0192 | 30.541 | 4.00 | 0.0644 | n/a |
| 10 | TRUE-NL | e2e | 20 | 20/20 | 0 | 1.6756 | 21.8444 | 29.817 | 4.00 | 0.0666 | n/a |
| 20 | LINEAR-EARLY-RATE | e2e | 20 | 17/20 | 3 | 2.0783 | 35.1400 | 45.579 | 5.65 | 0.2144 | n/a |
| 20 | LINEAR-SECANT | e2e | 20 | 19/20 | 1 | 2.1427 | 35.4165 | 45.448 | 5.79 | 0.2132 | n/a |
| 20 | TRUE-NL | e2e | 20 | 20/20 | 0 | 1.7619 | 33.7080 | 41.555 | 6.00 | 0.2229 | n/a |
| 40 | LINEAR-EARLY-RATE | e2e | 20 | 9/20 | 11 | 3.0609 | 57.1303 | 59.425 | 9.22 | 0.8372 | n/a |
| 40 | LINEAR-SECANT | e2e | 20 | 11/20 | 8 | 4.4422 | 64.0327 | 76.653 | 11.00 | 1.0306 | n/a |
| 40 | TRUE-NL | e2e | 20 | 20/20 | 0 | 3.3865 | 59.7172 | 67.356 | 10.75 | 0.8632 | n/a |

## Interpretation

Do not rank methods by feasible-only means when feasibility rates differ. `repair_TRUE-NL_after_*` is a separate repaired method, not the original linear plan. LINEAR-EARLY-RATE often plans under the surrogate but fails true-physics duration checks; LINEAR-SECANT often fails already under the planner on larger instances.
