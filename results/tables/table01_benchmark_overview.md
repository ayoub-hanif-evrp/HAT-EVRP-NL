# Montoya E-VRP-NL benchmark overview

- Objective convention: Full = travel+service+nonlinear charging; lit = travel+charging
- Units: hours / Wh
- Benchmark size: 10–320 customers
- Instance count: 120
- Methods / checkpoints: dataset only
- Search protocol: official XML (read-only)
- Seeds: n/a
- Source raw: `dataset/montoya-et-al-2017`

| n_customers | n_instances | battery_kWh | Tmax_h | note |
|---|---|---|---|---|
| 10 | 20 | 16.0 | 10.0 | layouts/stations vary by family |
| 20 | 20 | 16.0 | 10.0 | layouts/stations vary by family |
| 40 | 20 | 16.0 | 10.0 | layouts/stations vary by family |
| 80 | 20 | 16.0 | 10.0 | layouts/stations vary by family |
| 160 | 20 | 16.0 | 10.0 | layouts/stations vary by family |
| 320 | 20 | 16.0 | 10.0 | layouts/stations vary by family |

## Interpretation

Official Montoya testbed retained read-only. Station counts and charger mixes vary by tc0/tc1/tc2 families.
