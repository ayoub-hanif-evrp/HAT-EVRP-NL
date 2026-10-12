# Phase 1 + Cycle 2 research notes

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

## M3/M4 — Cycle 1 H1-H4 training and controlled ablation

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

## M5 — Cycle 1 learned CHARGE experiment

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

## Cycle 2 — richer history, interleaved DeepSets training, larger data

Architecture unchanged:
Shared Causal EV History Transformer → current-state fusion → DeepSets
`g = rho(sum(phi(EV_state_i + Transformer_hidden_i)))` → hierarchical
route/customer decisions → frvcpy nonlinear charging + independent check.

### Implementation changes

1. Visit tokens now encode arrival/departure time, arrival/departure SOC,
   travel energy since previous stop, charging amount/duration, node type,
   and service. Features are rebuilt from the optimized plan whenever frvcpy
   revises charging; customer-decision order stays separate.
2. Imitation trajectories interleave multiple open expert routes while
   preserving each route's customer order. Training logs show roughly
   37–40% multi-open decisions and ~60–70% competing-customer decisions.
3. Synthetic generator expanded (random/clustered/mixed customers;
   random and p-median-like stations; density and technology variation).
   Teachers: 80 train / 8 val feasible instances per run, independent seeds.
4. DeepSets diagnostics on H4_c2: `perm_error≈3e-8`, nonzero `phi_grad`,
   nonzero `active_logit_delta` when other EV states change.
5. Checkpoints and CSVs are versioned `*_c2*` and do not overwrite Cycle 1.

### Training (comparable across H1–H4)

```
python train.py --model H{k} --size 10 --steps 300 --rl-steps 30 \
  --train-instances 80 --val-instances 8 --eval-every 30 --seed {42,43,44} --version c2
```

Best validation BC/RL checkpoints saved under `checkpoints/H*_n10_seed*_c2.pt`.

### Official n=10 compare (20 instances, 10 samples; H0 equal-budget + H0_strong)

```
python evaluate.py --mode compare --size 10 --limit 20 --samples 10 \
  --seed {42,43,44} --version c2 --local-passes 3 --local-budget 4000
```

Per-seed mean verified objective (all 20/20 feasible):

| method     | seed 42   | seed 43   | seed 44   |
|------------|-----------|-----------|-----------|
| H0         | 22.157675 | 22.085038 | 21.945392 |
| H0_strong  | 20.368819 | 20.131997 | 19.924086 |
| H1         | 21.387641 | 20.541087 | 22.684219 |
| H2         | 21.108485 | 21.295244 | 20.740739 |
| H3         | 21.651773 | 20.651855 | 21.246115 |
| H4         | 21.429569 | 20.746494 | 20.698503 |

Pooled across three seeds (60 instance-seed rows), all 60/60 feasible:

- H0 22.062702; H0_strong 20.141634; H1 21.537649; H2 21.048156;
  H3 21.183247; H4 20.958189.
- H4 − H2 mean delta −0.0900 h; H4 better on only 28.3% of paired rows.
- H4 − H0 mean delta −1.1045 h; H4 better on 65.0% of paired rows.
- H4 − H0_strong mean delta +0.8166 h; H4 better on only 16.7%.

Interpretation: Cycle 2 reverses the Cycle 1 mean ranking of H4 vs H2 by a
small margin, but paired wins remain minority. DeepSets is active and
permutation-invariant, yet the coordination gain is weak on this static
benchmark. Equal-budget H0 is beaten by history models; strong H0+LS is not.

### Zero-shot scale

```
python evaluate.py --mode compare --size 20 --limit 20 --samples 5 \
  --seed 42 --version c2 --checkpoint-size 10 --local-passes 2 --local-budget 3000
```

n=20 (20/20 feasible): H0 36.690980; H0_strong 34.420101; H1 34.658276;
H2 34.594263; H3 34.658276; H4 34.579682. H4 is the best neural mean and
nearly matches H0_strong (gap 0.16 h).

```
python evaluate.py --mode compare --size 40 --limit 5 --samples 3 \
  --seed 42 --version c2 --checkpoint-size 10 --local-passes 2 --local-budget 3000
```

n=40 sample (5/5 feasible): H0 63.747925; H0_strong 60.168459; H1 62.057388;
H2 61.469290; H3 62.218225; H4 61.921243. History (H2) remains the best neural
mean; H4 is close but does not beat H2 or H0_strong.

### Cycle 2 direct CHARGE

- Physical absolute-Wh candidates from curve breakpoints, onward need, and
  percentage anchors; time-aware GO/CHARGE masks using exact curves.
- Retrained: `python train.py --model CHARGE --size 10 --steps 2000
  --train-instances 80 --seed 42 --version c2` (1236 expert actions).
- Eval: `python evaluate.py --mode direct --size 10 --limit 20 --samples 10
  --seed 42 --version c2`
- hybrid_H4: 20/20, mean 21.429569 h.
- direct_CHARGE: 4/20 feasible (same rate as Cycle 1); failures are
  stranded “no feasible GO or CHARGE” states after earlier decisions.
- direct_plus_repair: 20/20, mean 21.429569 h — repair is frvcpy, not learned.

### Artifacts

- Compare CSVs: `results/compare_size10_seed{42,43,44}_c2.csv`,
  `results/compare_size20_from10_seed42_c2.csv`
- Training curves: `results/training_H*_n10_seed*_c2.csv`,
  `results/learning_curves_c2.png`
- Direct: `results/direct_charge_size10_seed42_c2.csv`
- Checkpoints: `checkpoints/*_c2.pt` (Cycle 1 files preserved)

## Cycle 3 — final completion (`c3`)

### Investigation of Cycle 2

H4 vs H2 was seed-fragile (seed 42 mostly ties/losses; seed 43 wins). H4 opened
slightly more routes than H0_strong (≈3.07 vs 2.80) and was ~6–9× slower per
instance. DeepSets gradients and permutation invariance were healthy; the weak
H4−H2 gap is therefore not an implementation bug but limited coordination value
on this static offline problem.

### Efficiency

Cached static node embeddings, route-history keys, batched customer scoring, and
singleton oracle reuse. Smoke decode H4 on `tc0c10s2cf1`: ~0.042 s, 55 oracle
calls; LS refine reduced objective 29.02→25.28 with identical checker.

### Training

```
python train.py --model H{k} --size 10 --steps 400 --rl-steps 40 \
  --train-instances 120 --val-instances 12 --eval-every 40 --seed {42,43,44} --version c3
python train.py --model CHARGE --size 10 --steps 3000 --train-instances 120 --seed 42 --version c3
```

### Final n=10 (pooled 60)

Equal samples: H0 22.063; H0_strong 20.142; H1 21.249; H2 20.966; H3 22.131;
H4 20.975. H4−H2 +0.009 h (33% win). H4−H0_strong +0.833 h (12% win).

Equal LS (`--neural-local-passes 3 --neural-local-budget 4000`):
H1 19.644; H2 19.601; H3 19.702; H4 19.597 vs H0_strong 20.142.
H4−H0_strong −0.545 h (**76.7%** win). H4−H2 −0.004 h (near tie).

Wall-clock seed 42: H0 20.994; H0_strong 20.010; H2 20.792; H4 21.057.

### Scale (seed 42)

n=20 equal: H2 33.708; H4 33.820; H0_strong 34.420 (20/20).
n=20 eqls: H2 32.217; H4 32.307; H0_strong 34.420 (20/20).
n=40 eqls: H2 57.173; H4 57.506; H0_strong 60.195 (20/20).
n=80 eqls sample: H2/H4 88.151; H0_strong 95.506 (3/3). No full 160/320 run
on CPU.

### Direct CHARGE c3

Still 4/20 feasible; hybrid 20/20. Retained as negative ablation.

### Deliverables

`results/summary.md`, `paper_draft.md`, aggregate CSVs/figures `*_c3*`,
`example_solution_H4_c3_tc0c10s2cf1.json`, checkpoints `*_c3.pt`.

## Fairness correction (post Cycle 3 review)

Issue: legacy `eqls` applied LS to every neural sample (~2704 oracle/inst for
H4) while H0_strong used one LS after best-of-10 (~528). That comparison is
invalid for equal-effort claims.

Fix: `--fair-ls` = best-of-N constructions without LS, then one shared
`local_search` with identical passes/budget for all methods. Wall-clock uses
measured neural time per instance.

Fair n=10 pooled (60): H0_strong 20.142 / 528 oracle; H2 19.813 / 747;
H4 19.886 / 753. H2 and H4 beat H0_strong on 60% of pairs; H4 does not beat H2.
Fair wall-clock seed 42: H2 19.586 vs H0_strong 20.889 at ~0.48 s mean.
n=20/40 fairls: H2/H4 beat H0_strong with similar oracle counts (~1.1–1.2×).

Literature BKS (Froger 2021 Table 13): n=10 mean gap ≈41% (H2), 43% (H4).
DIMACS checker cloned; Java unavailable; sample MIP XML verified internally.

Legacy `eqls` CSVs retained but must not be cited as equal LS.

## Final conclusion

After correcting search effort, history-aware hybrid construction + one-shot LS
still improves on strong classical greedy+LS on n=10–40 in these runs, while
DeepSets shows no consistent benefit over history alone. Methods remain far from
published BKS. Direct learned charging remains a negative ablation.
