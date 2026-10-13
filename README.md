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

## Generate all figures and tables

```bash
python scripts/reporting/build_results.py --all
```

Outputs:

- [`results/README.md`](results/README.md) — definitive results index
- [`results/figures/`](results/figures/) — publication PNGs
- [`results/tables/`](results/tables/) — publication Markdown tables

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

## Documents

| Document | Role |
|---|---|
| [`paper_draft.md`](paper_draft.md) | Manuscript draft |
| [`results/README.md`](results/README.md) | Current vs historical results index |
| [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md) | Cycle-3 baseline tables |
| [`results/REGRESSION_RECOVERY_REPORT.md`](results/REGRESSION_RECOVERY_REPORT.md) | n40 recovery audit |
| [`docs/CLEANUP_AUDIT.md`](docs/CLEANUP_AUDIT.md) | Cleanup / retention audit |
