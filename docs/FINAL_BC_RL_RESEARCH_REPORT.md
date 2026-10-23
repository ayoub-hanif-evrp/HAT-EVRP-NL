# Final BC → RL research report

**Verdict: COMPLETE AND VERIFIED for the locked comparison.**

REINFORCE does not improve greedy construction. REINFORCE does improve the mean of four sampled constructions, at a matched inference budget, on the official n=10 and n=20 instances for the Route Transformer and for Route Transformer + DeepSets. That sampled result was predeclared on synthetic validation and then confirmed on the Montoya instances. It is not a better single-route deployment than greedy decoding.

The locked evaluation is commit `7671f6411a37666212656121d02a7c3af41ce1f7`, which was the tip of both `main` and `bc-rl-v1`. This report’s charging decomposition is the following commit, pushed to both branches. The distribution-KL rerun remains `6524326` and was not repeated. Historical result directories were not overwritten.

## What was tested

The policy learns route selection and customer order. `frvcpy` chooses stations and charge amounts. The independent checker accepts a route only when coverage, state of charge, duration, and the station charging time \(C_s(q_{\mathrm{out}})-C_s(q_{\mathrm{in}})\) all hold. Training remains behavior cloning, then REINFORCE. No third phase was added.

The new work is an evaluation audit of checkpoints that already existed:

- 20 original runs, four models, seeds 42–46, 1,000 BC updates and 300 RL updates, leave-one-out baseline, learning rate \(5\times 10^{-5}\). Manifest: `results/tables/final/original_run_manifest.csv`.
- 10 shuffled-order RL runs and 10 corrected distribution-KL runs, used only as a replication of the validation pattern.

Each checkpoint was scored on the same 100 synthetic validation instances in three ways: one greedy decode, the mean of four samples, and the best feasible sample among those four. The sample seeds are \(910000 + 16i + k\). Behavior cloning and REINFORCE use the same seeds. No local search is called.

The benefit rule was written down before the audit: at least 4 of 5 seeds better, and the paired 95% interval of RL minus BC entirely below zero. Greedy failed that rule for every model and every saved experiment. The mean of four samples and the best of four met it for every original model. The stopping rule then forbids another training configuration. The implemented greedy-rollout baseline at learning rate \(2\times 10^{-5}\) (`scripts/experiments/run_final_rl.py`) was not launched.

Official construction-only greedy numbers are the existing P0 evaluation at n=10, 20, 40, and 80. Official four-sample numbers cover all four models at n=10, n=20, and n=40. n=80 four-sample evaluation was not run.

## Evidence

Synthetic validation, original checkpoints, duration in hours. Negative means REINFORCE is lower. Feasibility is 1.0.

| Model | Protocol | BC | RL | Difference | Seeds | 95% interval |
|---|---|---:|---:|---:|---:|---|
| Route Transformer | Greedy | 21.475 | 21.516 | +0.040 | 3/5 | [−0.109, +0.190] |
| Route Transformer | Mean of 4 | 25.168 | 24.824 | −0.344 | 5/5 | [−0.528, −0.160] |
| Route Transformer | Best of 4 | 22.559 | 22.370 | −0.189 | 5/5 | [−0.267, −0.112] |
| RT + DeepSets | Greedy | 21.715 | 21.785 | +0.070 | 1/5 | [−0.048, +0.189] |
| RT + DeepSets | Mean of 4 | 25.544 | 24.587 | −0.958 | 5/5 | [−1.380, −0.536] |
| RT + DeepSets | Best of 4 | 22.864 | 22.247 | −0.616 | 5/5 | [−0.922, −0.310] |

Source: `results/tables/final/validation_summary.csv` and `results/raw/inference_audit_v1/decision.json`.

Official greedy Route Transformer, n=10: 22.379 h to 22.287 h, −0.091 h, interval [−0.336, +0.154], 3/5 seeds. No greedy cell at n=10, 20, 40, or 80 has an interval entirely below zero. Source: `results/tables/final/official_greedy_summary.csv`.

Official mean of four samples:

| Model | n | Difference (h) | Seeds | 95% interval |
|---|---:|---:|---:|---|
| Route Transformer | 10 | −0.714 | 5/5 | [−0.944, −0.484] |
| Route Transformer | 20 | −0.566 | 5/5 | [−0.965, −0.166] |
| RT + DeepSets | 10 | −1.432 | 5/5 | [−1.983, −0.882] |
| RT + DeepSets | 20 | −0.738 | 5/5 | [−1.328, −0.148] |

Best-of-four official intervals exclude zero at n=10 and include zero at n=20. At n=40 the Route Transformer mean of four samples is −1.251 h on 5/5 seeds, interval [−2.677, +0.176], which includes zero. Its best of four is −1.356 h, interval [−2.363, −0.348], 5/5. RT + DeepSets at n=40 includes zero for both protocols. Every sample in these cells is feasible. At n=10 the Route Transformer uses about 1,100 charging solves for greedy construction of the 20 instances and about 3,970 solves for four samples. The best REINFORCE sample (22.529 h) does not beat greedy behavior cloning (22.379 h).

The REINFORCE checkpoint was selected by greedy validation, not by the sampled score.

Linear charging plans on fixed routes are feasible under true Montoya physics on 16/20, 12/20, and 11/20 instances at n=10, 20, and 40. Exact nonlinear plans are feasible on 20/20. Those counts were recomputed from the instance files, not only from the summary. End-to-end true-physics feasibility at n=40 is 20/20, 11/20, and 9/20 for exact, secant, and early-rate planning. The two experiments use a Cycle-3 checkpoint. The final checkpoints were decoded again, without retraining, and the regenerated full durations match the locked official file. The sampled improvement is travel time. Charging time is a small reduction at n=10 and is not established at n=20. Details: `docs/FINAL_NONLINEAR_CHARGING_ANALYSIS.md`.

## What this does not show

It does not show that reinforcement learning improves the route a user gets from one greedy decode. It does not show that four samples should replace greedy decoding. It does not show that DeepSets improves the Route Transformer. It does not show a state-of-the-art gap to Froger et al. (2019); the Route Transformer n=10 literature gap stays above 22%. The comparison file is still named `literature_bks_froger2021.csv` and its source column says “Froger et al. 2021 Table 13” on every row. Checked n=10 and n=20 values match Tables 9 and 13 of the 2019 *Computers & Operations Research* article, not a 2021 paper, and they are not the later best-known solutions in Froger et al. (2022). The earlier sampled log-ratio penalty had no expected gradient. The later distribution penalty is a sampled-conditional KL, not the exact hierarchical KL, and it was not used to select these checkpoints. Shuffled sampling and that penalty reproduce the same validation split; they are not additional successes.

## Commands

Validation audit, from the repository root:

```text
python -u scripts/experiments/eval_inference_audit.py
```

Official four-sample evaluation at n=10 and n=20:

```text
python -u scripts/experiments/eval_locked_official.py --run-root results/raw/bc_rl_full_v1 --out results/raw/bc_rl_full_v1/eval_stochastic/official_samples.csv --models H1 H2 H3 H4 --seeds 42 43 44 45 46 --greedy-sizes 10 20 --sample-sizes 10 20
```

Tables and figures:

```text
python scripts/reporting/final_research_artifacts.py
python scripts/reporting/audit_nonlinear_instances.py
python scripts/reporting/charging_publication_figures.py
python scripts/reporting/summarize_route_decomposition.py
```

The route decomposition reads saved checkpoints and does not train:

```text
python -u scripts/experiments/eval_route_decomposition.py
```

Its output is `results/raw/bc_rl_route_decomposition_v1/`. Regenerated objectives matched the locked sample file on all 4,000 H2/H4 rows at n=10 and n=20.

Tests, after the manuscript and evaluation code were in place: 108 tests, OK. Transcript: `results/raw/final_test_transcript.txt`.

```text
python -m unittest discover -s tests
```

Environment for the tables in this report: Python 3.12.10, Windows 11, CPU. The audit and the official sample run used the same machine.

## Completion

| Required piece | Status |
|---|---|
| 20 original BC→RL runs, 300 RL updates | Present, not rerun |
| Greedy vs mean-sample vs best-of-four audit | Complete, 60 checkpoints, 100 instances |
| Further RL training | Not run; the predeclared rule was already met |
| Official greedy n=10, 20, 40, 80 | Present from the earlier construction-only evaluation |
| Official four-sample n=10, n=20, and n=40 | Complete, all feasible |
| Official four-sample n=80 | Not run |
| Official sample integrity | 12,000 rows, sha256 `05389d2aa990f8621c23439985babce9be96b220c16d6efad193d35f0f4827e9` |
| Final-checkpoint route decomposition | 4,000 rows, objectives matched, no retraining |
| Five-seed linear intervention on final weights | Not run |
| Manuscript | `paper_draft.md` |
| Figures | `results/final/figures/` |

**COMPLETE AND VERIFIED.** The verified positive statement is the mean of four samples at official n=10 and n=20. It is not established for that mean at n=40. The greedy statement is a negative result.
