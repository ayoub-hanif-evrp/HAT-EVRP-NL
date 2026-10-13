# Montoya E-VRP-NL benchmark overview

- Objective convention: Full = travel+service+nonlinear charging; lit = travel+charging
- Units: hours / Wh / km/h
- Benchmark size: 10–320 customers
- Instance count: 120
- Methods / checkpoints: dataset only
- Search protocol: official XML (read-only)
- Seeds: n/a
- Source raw: `dataset/montoya-et-al-2017`

| Customers | Instances | Stations (range) | Layout families | Charger types | Speed | Consumption | Battery kWh | Route limit h | Mean service h/cust |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 2–3 | tc0,tc1,tc2 | fast,normal,slow | 40 km/h (Montoya) | 125 Wh/km (Montoya) | 16.0 | 10.0 | 0.50 |
| 20 | 20 | 3–4 | tc0,tc1,tc2 | fast,normal,slow | 40 km/h (Montoya) | 125 Wh/km (Montoya) | 16.0 | 10.0 | 0.50 |
| 40 | 20 | 5–8 | tc0,tc1,tc2 | fast,normal,slow | 40 km/h (Montoya) | 125 Wh/km (Montoya) | 16.0 | 10.0 | 0.50 |
| 80 | 20 | 8–12 | tc0,tc1,tc2 | fast,normal,slow | 40 km/h (Montoya) | 125 Wh/km (Montoya) | 16.0 | 10.0 | 0.50 |
| 160 | 20 | 16–24 | tc0,tc1,tc2 | fast,normal,slow | 40 km/h (Montoya) | 125 Wh/km (Montoya) | 16.0 | 10.0 | 0.50 |
| 320 | 20 | 24–38 | tc1,tc2 | fast,normal,slow | 40 km/h (Montoya) | 125 Wh/km (Montoya) | 16.0 | 10.0 | 0.50 |

## Interpretation

Official Montoya testbed retained read-only. Speed/consumption follow the published Montoya parameterization used by the XML instances.
