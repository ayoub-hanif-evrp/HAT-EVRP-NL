# Final reproducibility audit

The locked evaluation is commit `7671f6411a37666212656121d02a7c3af41ce1f7`. The charging decomposition and the reviewer statistics are the commits after it. Commands below are relative to the repository root. They do not use a personal machine path.

Library versions for the reviewer statistics are in `results/tables/final/reproducibility_manifest.json`. The statistics and the slow-charger summary do not retrain:

```text
python scripts/reporting/peer_review_statistics.py
python -u scripts/experiments/eval_shift_holdout.py
python scripts/reporting/summarize_shift_holdout.py
```

The holdout protocol is `results/raw/shift_holdout_v1/frozen_protocol.json`. It was written before the first decode. The budget was 480 s. The run finished in 39 s. Checkpoints were not reselected.

Python dependencies are the packages already imported by the training and reporting scripts: PyTorch, NumPy, Matplotlib, and `frvcpy`. The machine used for the new evaluation was Python 3.12.10 on Windows 11, CPU.

## Locked official sample file

`scripts/reporting/final_research_artifacts.py` calls `validate_official_samples` before it writes tables. The check fails if any of the following is wrong:

- 12,000 rows.
- Models H1–H4, seeds 42–46, roles `bc` and `rl`, sizes 10, 20, and 40.
- Twenty instances in every group.
- One greedy trial and four sample trials, with no duplicate key.
- The same instance identifiers for BC and RL.
- Finite objectives, feasible flags, and oracle requests at least as large as oracle executions.

The file `results/raw/bc_rl_full_v1/eval_stochastic/official_samples.csv` passed. SHA-256: `05389d2aa990f8621c23439985babce9be96b220c16d6efad193d35f0f4827e9`. The raw file was not edited.

## Route decomposition

```text
python -u scripts/experiments/eval_route_decomposition.py
python scripts/reporting/summarize_route_decomposition.py
```

The decoder uses the same instance order and the same sample seeds as `scripts/experiments/eval_locked_official.py`: seed \(910000 + 16i + k\). A regenerated objective that differs from the locked file by more than \(10^{-4}\) h stops the script. The completed run has 4,000 rows, all errors are 0, and all duration identities hold. Raw rows: `results/raw/bc_rl_route_decomposition_v1/solutions.csv` (SHA-256 `2549c56d10df86ef794f00ee2a7abecb59e880a61b41bda437923e49b2cf48a3`) and `charges.csv`. Summary: `results/tables/final/route_decomposition_summary.csv`.

## Charging counts and figures

```text
python scripts/reporting/audit_nonlinear_instances.py
python scripts/reporting/charging_publication_figures.py
python scripts/reporting/final_research_artifacts.py
```

Instance counts in `results/tables/final/nonlinear_instance_audit.csv` were computed from `fixed_route_charging_comparison.csv` and `end_to_end_charging_sensitivity.csv`. They match `charging_study_summary.csv`. Figures read those tables, the saved schedule JSON, or the decomposition summary. No figure invents a curve or a route.

## Best-known solutions

`results/literature_bks_froger2021.csv` keeps its historical source string, “Froger et al. 2021 Table 13,” on every row. That string is not a correct uniform citation. Checked values match Froger, Mendoza, Jabali, and Laporte, *Computers & Operations Research* 104 (2019), doi:10.1016/j.cor.2018.12.013: `tc0c10s2cf1` = 19.75 h is in Table 9, and `tc0c20s3ct2` = 17.08 h is in Table 13. Froger, Jabali, Mendoza, and Laporte, *Transportation Science* 56 (2022), doi:10.1287/trsc.2021.1111, reports later best-known solutions. Those later values are not this file. The CSV was not rewritten.

## Tests

```text
python -m unittest discover -s tests
```

The transcript is `results/raw/final_test_transcript.txt`, written as UTF-8. CSV reads in the tests that previously left files open now use context managers.

## Not run

No new BC or RL training. `scripts/experiments/run_final_rl.py` was not launched. No five-seed linear-planner intervention on the final checkpoints. n=80 four-sample evaluation was not run.
