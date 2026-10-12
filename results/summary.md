# HAT-EVRP-NL summary (points to final report)

**See [`FINAL_REPORT.md`](FINAL_REPORT.md) for authoritative numbers and verdict.**

Closure fixes: classical wall-clock RNG (`deterministic_first`); fair one-shot
LS retained; Java checker skipped; BKS from Froger et al. (2021) Table 13.

Final fair LS n=10 (20 instances, seeds averaged in paired tests): H2 19.813 h,
H4 19.886 h, H0_strong 20.142 h. **H4 does not improve over H2.** Wall-clock
with diverse classical constructions: H2 19.586, H4 19.840, H0_strong 19.895
(seed 42). BKS gaps ~43%. Direct CHARGE 4/20 feasible.
