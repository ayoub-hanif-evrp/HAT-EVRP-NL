# HAT-EVRP-NL

Research code for **History-Aware Transformer + DeepSets** hybrid routing on
Montoya E-VRP-NL with exact nonlinear charging (`frvcpy`).

**Authoritative conclusions:** [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md)  
**Paper draft:** [`paper_draft.md`](paper_draft.md)  
**Paired stats:** [`results/FINAL_paired_by_instance.csv`](results/FINAL_paired_by_instance.csv)

## Install

```bash
pip install -r requirements.txt
python evaluate.py --mode sanity --data dataset
```

Validation is **independent Python forward simulation** (`check_route` /
`check_solution`). No Java or external DIMACS checker is required.

## Load best Cycle-3 checkpoints

Checkpoints: `checkpoints/{H1|H2|H3|H4}_n10_seed{42|43|44}_c3.pt`  
H2 is the strongest efficient neural variant; H4 is the full architecture.

```python
from train import load_policy, decode
from data import parse_instance
policy = load_policy("checkpoints/H2_n10_seed42_c3.pt")
```

## Reproduce final comparisons (existing `*_c3.pt` only; no retraining)

```bash
# A. Equal constructions (no LS on neural)
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

## Verdict (short)

Transformer history helps. DeepSets does not reliably beat history-only (**H2**).
Fair LS (n=10): H2 19.813, H4 19.886, H0_strong 20.142; H4−H2 20%/60%/20%.
Hybrid `frvcpy` charging is required; direct CHARGE is 4/20 feasible. Large gaps
remain vs Froger et al. (2021) BKS (~43%).
