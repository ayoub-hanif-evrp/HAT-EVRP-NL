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
  Montoya node set. A depot copy now participates only in lower-bound
  calculations and is explicitly removed from the insertable-station set.

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

## M2 strengthened — full H0 evaluation

- Corrected a second adapter issue found on `tc0c40s5ct0`: making the terminal
  helper unreachable incorrectly rejected customers that could return directly
  to the depot. Singleton routes for customers 24 and 32 now independently
  verify at 3.429112 h and 3.286251 h.
- Added oracle-verified best-improvement 2-opt and relocate search.
- Commands:
  - `python evaluate.py --mode baseline --size 10 --limit 20 --restarts 10 --local-passes 3 --local-budget 4000 --seed 42`
  - `python evaluate.py --mode baseline --size 20 --limit 20 --restarts 10 --local-passes 3 --local-budget 5000 --seed 42`
  - `python evaluate.py --mode baseline --size 40 --limit 5 --restarts 5 --local-passes 2 --local-budget 5000 --seed 42`
- H0 results:
  - n=10: 20/20 feasible, mean/median 20.368819/18.900300 h, 1.625 s total.
  - n=20: 20/20 feasible, mean/median 32.396377/31.160697 h, 10.821 s total.
  - n=40: 5/5 feasible, mean/median 59.954062/59.278753 h, 8.029 s total.

## M3/M4 — H1-H4 training and controlled ablation

- CPU-only environment: PyTorch 2.14.0, no CUDA.
- Architecture: d=128, two shared causal Transformer layers, four heads,
  explicit optimized-prefix state, optional DeepSets, hierarchical
  route/new-route and customer/close pointers, exact frvcpy masks and final
  independent validation.
- Synthetic data are independently generated random/clustered/mixed instances.
  Thirty feasible n=10 teachers per model came from H0 + charging optimization.
- Commands used `--steps 200 --rl-steps 20 --train-instances 30
  --val-instances 5 --eval-every 20 --seed 42` for each H1-H4.
- Best validation checkpoints are retained. REINFORCE degraded H1 and H3, so
  their best BC weights were kept; it improved H2 and H4 slightly.
- Controlled official n=10 command:
  `python evaluate.py --mode compare --size 10 --limit 20 --samples 10 --seed 42`
- All methods: 20/20 feasible. Mean verified objective:
  H0 22.157675 h; H1 22.911023 h; H2 21.013877 h; H3 22.911023 h;
  H4 21.608789 h.
- History helped in this run (H2 was 5.16% below H0), but DeepSets did not:
  H4 was 2.83% worse than H2 and H3 matched H1.
- Repeated the complete 200-BC/20-RL training and 20-instance evaluation with
  seed 43. Means H0/H1/H2/H3/H4 were 22.085038/21.088446/20.869668/
  21.186288/21.379565 h; all were 20/20 feasible.
- Across both seeds (40 instance-seed observations), means were
  22.121357/21.999735/20.941772/22.048655/21.494177 h. H2 improved over H0
  by 5.33%; H4 improved over H0 by 2.84% but remained 2.64% worse than H2.
  The full history+DeepSets hypothesis is therefore not supported.
- This controlled H0 excludes local search to match ten route-construction
  samples. The stronger H0 with local search averaged 20.368819 h on the same
  20 instances, so even H2 remained 2.81% worse than the strongest classical
  result. Neural superiority over H0 is not established.
- Zero-shot n=10-to-n=20 on five official instances, five samples:
  H0/H1/H2/H3/H4 means were 39.063000/36.159551/35.124563/36.179136/
  36.179136 h; every solution was feasible.

## M5 — learned CHARGE experiment

- Trained a secondary GO/CHARGE, station-pointer, target-SOC policy for 2,000
  updates on 463 oracle actions. Exact nonlinear curves are always used.
- Added necessary reachability and onward-energy masks after diagnosing
  stranded vehicles; no output actions are retrospectively altered.
- Command:
  `python evaluate.py --mode direct --size 10 --limit 20 --samples 10 --seed 42`
- Hybrid H4: 20/20 feasible, mean 21.608789 h.
- Direct learned charging: 4/20 feasible. Mean 20.050314 h over only those four
  successes is not comparable and must not be interpreted as better.
- Direct+repair: 20/20 feasible, mean 21.608789 h. This exactly reflects
  replacement by the charging optimizer, not quality of learned station heads.

## Current conclusion and limitation

The causal history representation was beneficial across two seeds, while the
DeepSets hypothesis is not supported. The direct learned charging policy is
far less reliable than hybrid optimization. These are small CPU experiments
without local BKS values; two training seeds are still insufficient for a
strong statistical claim. The next justified experiment is a third seed and
larger-instance training before architecture tuning.
