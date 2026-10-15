# HAT-EVRP-NL

Hybrid **History-Aware Transformer + DeepSets** construction for the electric
vehicle routing problem with **nonlinear charging** (Montoya E-VRP-NL), using
exact fixed-route charging optimization (`frvcpy`) and an independent Python
feasibility validator.

## Motivation

E-VRP-NL couples customer sequencing with piecewise-linear concave charging.
This repository studies whether causal EV-history encoding and DeepSets fleet
aggregation improve neural construction when charging amounts and station
visits are optimized by an exact FRVCP solver for each fixed customer order.

## Objective (E-VRP-NL)

For each feasible solution:

- **Full duration** = travel + service + nonlinear charging
- **Literature-compatible duration** = travel + nonlinear charging  
  (= full − Σ customer service, constant once all customers are covered)

Nonlinear charging time for a visit is \(F_s(q_\text{out}) - F_s(q_\text{in})\).
Service time counts toward the route-duration limit \(T_{\max}\) but is excluded
from literature BKS gaps. `frvcpy` optimizes charging for a **fixed** customer
sequence; it does not prove global VRP optimality.

## Architecture and ablations

Shared causal EV-history Transformer → dynamic route-state fusion → DeepSets
fleet aggregation → hierarchical route/customer decoder → `frvcpy` → Python
`check_route` / `check_solution`.

| Variant | Meaning |
|---|---|
| H0 / H0_strong | Classical baselines |
| H1 | State-only neural |
| H2 | + history Transformer |
| H3 | DeepSets without history |
| H4 | Full architecture |

## Install

```bash
pip install -r requirements.txt
python evaluate.py --mode sanity --data dataset
```

No Java and no external checker installation are required.

## Minimal inference

```python
from train import load_policy, decode
from data import parse_instance

policy = load_policy("checkpoints/H2_n10_seed42_c3.pt")
instance = parse_instance("dataset/montoya-et-al-2017/tc0c10s2cf1.xml")
result = decode(policy, instance, sample=False, seed=42)
```

## Reproduce main benchmarks

```bash
# Cycle-3 fair-LS (n=10)
python evaluate.py --mode compare --size 10 --limit 20 --samples 10 --seed 42 \
  --version c3 --local-passes 3 --local-budget 4000 --fair-ls --tag fairls_final

# Recovered n=40 ordinary λ=0 + hybrid search
python scripts/experiments/run_regression_recovery.py --stage pipeline
```

## Research questions

1. **RQ1** — Does causal route history improve over state-only neural routing?
2. **RQ2** — Does DeepSets improve upon the history Transformer?
3. **RQ3** — Does nonlinear charging matter vs linear surrogates under true physics?
4. **RQ4** — Does REINFORCE improve upon behavior cloning?

## Nonlinear charging study (RQ3)

```bash
python scripts/experiments/run_nonlinear_charging_study.py --stage all
python scripts/reporting/charging_and_reward_figures.py
```

Report: [`results/NONLINEAR_CHARGING_STUDY.md`](results/NONLINEAR_CHARGING_STUDY.md) · Table: [`results/tables/table15_nonlinear_vs_linear_charging.md`](results/tables/table15_nonlinear_vs_linear_charging.md)

## Controlled BC vs RL (RQ4; does not overwrite Cycle-3)

```bash
python scripts/experiments/run_bc_rl_controlled.py --stage all --models H2,H4 --seeds 42,43,44
```

Report: [`results/REWARD_AND_LEARNING_STUDY.md`](results/REWARD_AND_LEARNING_STUDY.md)

## Generate all figures and tables

```bash
python scripts/reporting/publication_figures.py   # 20 main-paper PNGs
python scripts/reporting/build_results.py --all --skip-decomp
```

- [`results/figures/`](results/figures/) — 20 main-paper PNGs ([`docs/FIGURE_CATALOG.md`](docs/FIGURE_CATALOG.md))
- [`results/supplementary/figures/`](results/supplementary/figures/) — supporting plots + contact sheet
- [`results/tables/`](results/tables/) — Markdown tables
- [`paper_draft.md`](paper_draft.md) — manuscript with embedded figures

## Current strongest supported results

| Setting | Configuration | Mean full h |
|---|---|---:|
| n=40 | H4 Cycle-3 + ordinary λ=0 + hybrid_insert | **56.370** |
| n=40 | H2 Cycle-3 + ordinary λ=0 + hybrid_insert | **56.806** |
| n=10 | H2 Cycle-3 fair-LS (seeds 42–44) | **19.813** |

Historical fair-LS n40 reference: H2 57.173 h, H4 57.506 h.

Corrected literature BKS gaps (travel+charging vs Froger Table 13) are on the
order of **~5%** for H2 on n=10 fair-LS — **not** the obsolete ~43% figure that
compared full duration to literature BKS.

## Important limitations

- H4’s better recovered n40 mean (single seed) does **not** establish a general DeepSets advantage.
- Equal wall-clock neural-search superiority has **not** been established.
- Recent official-benchmark search results are exploratory.
- Charging optimization is exact only for the fixed-route subproblem.

## Directory overview

```text
data.py, model.py, train.py, evaluate.py, charging.py, solver.py, solver_ls.py
dataset/                 # official Montoya XML (read-only)
checkpoints/             # Cycle-3 primary + ablation + historical
scripts/experiments/     # experiment drivers
scripts/reporting/       # build_results.py
results/figures|tables|raw|archive
docs/CLEANUP_AUDIT.md
paper_draft.md
```

## Method display names

| Key | Name |
|---|---|
| H0 | Randomized Greedy Routing |
| H0_strong | Randomized Greedy + Local Search |
| H1 | State-Only Hierarchical Pointer |
| H2 | History Transformer Routing |
| H3 | DeepSets Fleet Routing |
| H4 | History Transformer + DeepSets (HAT-EVRP-NL) |

## Learning note

Cycle-3 uses **400 BC + 40 standard REINFORCE** on synthetic n=10. Historical synthetic-val RL endpoints beat best-BC on **1/3 H2** and **0/3 H4** seeds; deployed H4 often retains BC-selected weights. A separate controlled RQ4 bake-off (construction-only official n=10) shows H4 RL improving on 3/3 seeds and H2 on 1/3 — protocol-dependent, not a claim that RL always helps deployed Cycle-3 H4. Details: [`docs/TRAINING_AND_REWARD_METHODOLOGY.md`](docs/TRAINING_AND_REWARD_METHODOLOGY.md), [`results/REWARD_AND_LEARNING_STUDY.md`](results/REWARD_AND_LEARNING_STUDY.md).

## Documents

| Document | Role |
|---|---|
| [`paper_draft.md`](paper_draft.md) | Manuscript (RQ1–RQ4) |
| [`docs/METHODOLOGY_AUDIT.md`](docs/METHODOLOGY_AUDIT.md) | RQ1–RQ4 strengths, confounders, required ablations |
| [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) | Reproducible commands and compute policy |
| [`docs/PHASE1_CHANGE_REPORT.md`](docs/PHASE1_CHANGE_REPORT.md) | Correctness fixes change log |
| [`results/SCIENTIFIC_RESULTS_AUDIT.md`](results/SCIENTIFIC_RESULTS_AUDIT.md) | Independent numerical audit |
| [`results/NONLINEAR_CHARGING_STUDY.md`](results/NONLINEAR_CHARGING_STUDY.md) | Linear vs nonlinear under true physics |
| [`results/REWARD_AND_LEARNING_STUDY.md`](results/REWARD_AND_LEARNING_STUDY.md) | BC/RL + reward logs |
| [`results/README.md`](results/README.md) | Results index |
| [`docs/TRAINING_AND_REWARD_METHODOLOGY.md`](docs/TRAINING_AND_REWARD_METHODOLOGY.md) | Training + reward specification |
| [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md) | Cycle-3 baseline tables |
| [`results/REGRESSION_RECOVERY_REPORT.md`](results/REGRESSION_RECOVERY_REPORT.md) | n40 recovery audit |
