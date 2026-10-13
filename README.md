# HAT-EVRP-NL

Frozen research code: **History-Aware Transformer + DeepSets** hybrid routing on
Montoya E-VRP-NL with exact nonlinear charging (`frvcpy`).

| Document | Role |
|---|---|
| [`paper_draft.md`](paper_draft.md) | Full research manuscript |
| [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md) | Authoritative numerical tables |
| [`results/FINAL_paired_by_instance.csv`](results/FINAL_paired_by_instance.csv) | Protocol-labeled paired stats |
| [`results/figures/`](results/figures/) | Final manuscript figures |

## Install

```bash
pip install -r requirements.txt
python evaluate.py --mode sanity --data dataset
```

Validation is **independent Python forward simulation** (`check_route` /
`check_solution`). No Java or external DIMACS checker is required.

Dependencies: see `requirements.txt` (PyTorch, `frvcpy`, NumPy, matplotlib for plots).

## Load Cycle-3 checkpoints

```text
checkpoints/{H1|H2|H3|H4}_n10_seed{42|43|44}_c3.pt
```

H2 is the strongest efficient neural variant; H4 is the full architecture.

```python
from train import load_policy, decode
from data import parse_instance

policy = load_policy("checkpoints/H2_n10_seed42_c3.pt")
instance = parse_instance("dataset/montoya-et-al-2017/tc0c10s2cf1.xml")
result = decode(policy, instance, sample=False, seed=42, local_passes=3, local_budget=4000)
```

## Reproduce fair benchmarks (existing `*_c3.pt`; no retraining)

```bash
# A. Equal constructions (neural without LS; H0_strong still uses LS)
python evaluate.py --mode compare --size 10 --limit 20 --samples 10 --seed 42 \
  --version c3 --local-passes 3 --local-budget 4000

# B. Fair one-shot LS (best-of-N, then one shared local search)
python evaluate.py --mode compare --size 10 --limit 20 --samples 10 --seed 42 \
  --version c3 --local-passes 3 --local-budget 4000 --fair-ls --tag fairls_final

# C. Fair wall-clock (measured neural time; fixed classical RNG)
python evaluate.py --mode compare --size 10 --limit 20 --samples 10 --seed 42 \
  --version c3 --local-passes 3 --local-budget 4000 --fair-ls --wallclock --tag wallclock_final
```

Do **not** cite legacy `*_eqls_*` or `*_wallclock_fair_*` as equal-effort results
(see `results/INVALID_wallclock_fair_seed42_c3.md`).

## Layout

`data.py` · `charging.py` · `solver.py` · `model.py` · `train.py` · `evaluate.py`  
`checkpoints/*_c3.pt` · `results/` · `dataset/` (read-only)

## Verified principal findings (Cycle-3 baseline)

- Fair LS n=10: H0_strong **20.142**, H1 **19.993**, **H2 19.813**, H3 **20.109**, H4 **19.886** h.
- H4−H2: **20% / 60% / 20%** (no DeepSets win). History helps (H2≪H1).
- Hybrid `frvcpy` required; direct CHARGE **4/20** feasible.
- **Corrected** lit BKS gaps (travel+charging vs Froger Table 13): H2 **~5%**, not ~43% (service-time convention). See `results/IMPROVEMENT_REPORT.md`.

## Optional improved inference (post-freeze; same checkpoints)

```bash
# Oracle-aware pointer scoring (H2: λ=2, H4: λ=4) + fair LS
python evaluate.py --mode compare --size 10 --limit 20 --samples 10 --seed 42 \
  --version c3 --fair-ls --tag costaware --cost-lambda 2 --methods H2
```

Strong LS: `local_search(..., operators="strong")` under the same evaluation budget.  
Details: [`results/IMPROVEMENT_REPORT.md`](results/IMPROVEMENT_REPORT.md).
