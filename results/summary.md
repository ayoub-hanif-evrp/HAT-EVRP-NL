# HAT-EVRP-NL executed research summary

All reported solutions were checked by an independent forward simulator using
the parsed Montoya piecewise charging curves, service times, full initial
battery, SOC bounds, customer coverage, depot return, and 10-hour route limit.
No local BKS files were available, so optimality gaps are `NA`.

## Main n=10 ablation

Twenty official instances, seed 42, ten constructed solutions per method:

- H0 improved randomized greedy: 20/20 feasible, mean 22.157675 h.
- H1 state-only neural pointer: 20/20 feasible, mean 22.911023 h.
- H2 causal history without DeepSets: 20/20 feasible, mean 21.013877 h.
- H3 state-only with DeepSets: 20/20 feasible, mean 22.911023 h.
- H4 history plus DeepSets: 20/20 feasible, mean 21.608789 h.

The complete training/evaluation was repeated with seed 43. Across both seeds
(40 instance-seed observations), H0/H1/H2/H3/H4 means were 22.121357,
21.999735, 20.941772, 22.048655, and 21.494177 h. H2 improved over H0 by
5.33%. H4 improved over H0 by 2.84% but was 2.64% worse than H2. These runs
support useful customer-decision history, but not an additional benefit from
the current DeepSets fleet aggregation. Two seeds remain insufficient for a
strong statistical claim.

For budget control, H0 above used ten constructions without local search. The
stronger local-search H0 averaged 20.368819 h on the same 20 instances.
Therefore H2 was still 2.81% worse than the strongest classical result; the
experiments do not establish neural superiority over improved H0.

## Scale and generalization

Improved H0 was feasible on 20/20 n=10, 20/20 n=20, and 5/5 sampled n=40
instances. Mean objectives were 20.368819, 32.396377, and 59.954062 h.

Without retraining, n=10 neural checkpoints remained 5/5 feasible on sampled
n=20 instances. H2 had the best mean objective (35.124563 h) versus H0
39.063000 h under five constructions per method.

## Learned charging

The direct GO/CHARGE/station/target-SOC policy was feasible on only 4/20 n=10
instances despite expert imitation and necessary physical masks. Hybrid H4 was
20/20 feasible. Post-optimizing the direct policy's customer orders restored
20/20 feasibility, but that gain belongs to `frvcpy`, not the learned charging
heads.

## Reproduction

Key commands are recorded in `results/notes.md`. Checkpoints are in
`checkpoints/`; per-instance CSVs, learning curves, and a fully concrete
feasible route example are in `results/`.
