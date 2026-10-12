# Phase 1 research notes

Date: 2026-10-12

## M0 — dataset understood

- Command: `python evaluate.py --mode inspect --data dataset`
- Found 120 XML instances: 20 each with 10, 20, 40, 80, 160, and 320
  customers. The folder also contains `readme.txt`; no solutions or BKS files
  are present.
- Confirmed units and constants: coordinates in km; unrounded double-precision
  Euclidean distances; 40 km/h; 125 Wh/km; 16,000 Wh battery; 10 h route
  duration; 0.5 h customer service.
- Curves are cumulative energy-versus-time piecewise-linear functions for
  slow, normal, and fast station technologies (only the types used by an
  instance are included).
- Output: `results/dataset_summary.csv`.

## M1 — verified core

- Commands:
  - `python -m unittest frvcpy.test.test -v`
  - `python evaluate.py --mode sanity --data dataset --instance tc0c10s2cf1`
- Installed environment: Python 3.12.10, frvcpy 0.1.1, Windows 11, Intel64
  Family 6 Model 183.
- The three upstream frvcpy reference tests pass.
- Our travel/energy matrices agree with frvcpy's translator within
  1.82e-12 Wh (a `hypot` versus `sqrt` ULP difference).
- Verified customer route `(1,)` independently:
  duration 5.845844662 h, charging 0.219541863 h, one charging stop, terminal
  SOC 0 Wh. Concrete path: depot 0, customer 1, station 12 charging
  9631.513992 Wh, depot 0.
- Tightening the route limit to 5.845834662 h correctly rejects that route.
- Adapter correction: frvcpy otherwise requires a post-destination charger.
  Its stock translator adds a usable depot charger, which is not part of the
  Montoya node set. We instead add an inaccessible terminal helper so the
  destination bound is correct without permitting mid-route depot charging.

## M2 — randomized-greedy baseline

- Commands:
  - `python evaluate.py --mode baseline --data dataset --size 10 --limit 5 --restarts 5 --seed 42`
  - `python evaluate.py --mode baseline --data dataset --size 20 --limit 5 --restarts 5 --seed 42`
- Every reported solution passed the independent full-coverage, SOC, charging,
  service-time, and route-duration checks.
- 10 customers: 5/5 feasible; mean objective 22.760515 h; median 22.140444 h;
  mean 2.8 routes; mean 4.0 charging stops; mean runtime 0.036 s.
- 20 customers: 5/5 feasible; mean objective 37.360496 h; median 32.947810 h;
  mean 4.8 routes; mean 6.0 charging stops; mean runtime 0.192 s.
- Per-instance results and customer routes:
  `results/baseline_size10_seed42.csv` and
  `results/baseline_size20_seed42.csv`.
- Limitation: these are absolute objectives from a small deterministic sample;
  no local verified BKS is available, so no optimality gaps are reported.

## Next highest-value experiment

Add a compact insertion/relocate improvement to H0 and evaluate all 20
10-customer instances with the same seed and runtime accounting. This will
produce a stronger stable teacher/reference baseline before any neural model is
implemented.
