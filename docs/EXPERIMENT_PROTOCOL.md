# Experiment Protocol — HAT-EVRP-NL

Executable commands, seed policy, metrics, and approximate compute.
**Do not launch expensive multi-seed sweeps without authorization.**

## Environment record (fill at run time)

| Field | Value |
|---|---|
| Repo commit | `git rev-parse HEAD` |
| Python | `python --version` |
| PyTorch | `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` |
| frvcpy | `python -c "import frvcpy; print(getattr(frvcpy,'__version__','unknown'))"` |
| OS / CPU / RAM | |
| Seeds | training `{42,43,44}` minimum; prefer `{42..46}` or `{42..51}` |

## Metrics (every confirmatory row)

- `feasible` (0/1)
- `objective_h` = \(Z_\mathrm{full}\)
- `lit_h` = \(Z_\mathrm{lit}\); `bks_gap_lit_pct`
- `runtime_s` (wall-clock)
- `oracle_solver_executions`, `oracle_requests`, `oracle_cache_hits`
- `n_routes`, `n_charges`, `charging_h`
- `training_seed`, `eval_seed` (if distinct), `instance`, `protocol`

## Protocol blocks (never mix in one ranking)

| ID | Name | Description |
|---|---|---|
| P0 | Construction-only | Decode + frvcpy; no LS/repair |
| P1 | Fair one-shot LS | Matched LS passes/budget after best-of-K construct |
| P2 | Hybrid insert/repair | Classical + neural insertion (expensive) |
| P3 | Wall-clock classical | `_classical_until_time` with measured deadline |

## Smoke tests (cheap — run first)

```bash
python -m unittest tests.test_phase1_correctness tests.test_repair_and_validation -q
python -m unittest tests.test_publication_figures tests.test_learning_presentation -q
python scripts/analysis/paired_stats.py results/raw/cycle3/compare_size10_fairls_final_seed42_c3.csv --a H2 --b H4
```

Approximate: < 2 minutes CPU (excluding full figure regen).

## Lightweight validation (optional)

```bash
# Single-instance construction decode (requires checkpoint)
python -c "from train import load_policy, decode; from data import parse_instance, instance_files; from pathlib import Path
inst=parse_instance([p for p in instance_files(Path('dataset')) if 'c10' in p.name][0])
m=load_policy('checkpoints/H2_n10_seed42_c3.pt')
print(decode(m, inst, sample=False, seed=0)['objective'])"
```

## Confirmatory evaluation (expensive — authorize first)

Manifest sketch (`results/raw/experiment_manifest.yaml` to be written at run):

```yaml
name: confirmatory_fairls_n10_n20_n40
methods: [H0_strong, H1, H2, H3, H4]
sizes: [10, 20, 40]
protocol: P1_fair_ls
training_seeds: [42, 43, 44]   # extend to 5–10 if feasible
instances: official_montoya
freeze_before_test: true
approx_cpu_hours: "estimate after smoke; n40 hybrid is dominant cost"
```

Example command pattern (do not run blindly):

```bash
python evaluate.py compare --size 10 --fair-ls --local-operators basic \
  --version c3 --seed 42 --tag confirmatory
```

n=80 only if ≥ enough instances for layout CIs (current thin 3-instance results are exploratory only).

## Training-data scaling (planned)

Compare existing synthetic BC size vs larger generators under matched steps/seeds.
Freeze architecture and evaluation protocol before scaling runs.

## Statistical analysis

```bash
python scripts/analysis/paired_stats.py <per_instance_csv> --a H2 --b H4
```

Uses layout-mean deltas and descriptive 1.96·SE intervals over layouts.

## Reproducibility guards

- Never overwrite `checkpoints/*` historical files; write new version tags.
- Never modify `dataset/**/*.xml`.
- Write new raw CSVs under `results/raw/<study>/`; regenerate tables/figures from them.
- Record commit SHA in every study README.
