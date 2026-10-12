# SUPERSEDED — invalid wall-clock comparison

File: `compare_size10_wallclock_fair_seed42_c3.csv`

**Bug:** `_classical_until_time` called `randomized_greedy(restarts=1)`, and
`randomized_greedy` always took the deterministic greedy branch for
`restart == 0`. Every single-restart trial therefore produced the **same**
construction regardless of seed.

**Status:** Invalid for scientific claims. Replaced by
`compare_size10_wallclock_final_seed42_c3.csv` after fixing
`deterministic_first` in `solver.py`.
