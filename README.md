# HAT-EVRP-NL

Research code for **History-Aware Transformer + DeepSets** hybrid routing on
Montoya E-VRP-NL with exact nonlinear charging (`frvcpy`).

**Authoritative conclusions:** [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md)  
**Paper draft:** [`paper_draft.md`](paper_draft.md)

## Install

```bash
pip install -r requirements.txt
python evaluate.py --mode sanity --data dataset
```

## Reproduce final comparisons (use existing `*_c3.pt` checkpoints)

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

Transformer history helps. DeepSets does not reliably beat history-only (H2).
Hybrid `frvcpy` charging is required; direct CHARGE is 4/20 feasible. Large gaps
remain vs Froger et al. (2021) BKS.
