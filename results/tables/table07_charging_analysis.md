# Nonlinear charging analysis (n=40 recovered hybrid)

- Objective convention: Charging component of full duration
- Units: hours / kWh
- Benchmark size: 40
- Instance count: 20
- Methods / checkpoints: H2/H4
- Search protocol: same as table02
- Seeds: 42
- Source raw: `RECOVERY_n40_objective_decomposition.csv`

| Method | Mean charge h | Mean energy kWh | Mean visits | Identity OK rate |
|---|---|---|---|---|
| H2 | 3.0601 | 60.786 | 9.85 | 100% |
| H4 | 3.1511 | 60.434 | 9.65 | 100% |

## Interpretation

Identity OK means travel+service+charge matches full duration within 1e-6 h.
