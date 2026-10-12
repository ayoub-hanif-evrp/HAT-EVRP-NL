# HAT-EVRP-NL executed research summary

All reported solutions were checked by an independent forward simulator using
the parsed Montoya piecewise charging curves, service times, full initial
battery, SOC bounds, customer coverage, depot return, and 10-hour route limit.
No local BKS files were available, so optimality gaps are `NA`.

## Cycle 2 main result (n=10, three seeds)

Architecture preserved: shared causal visit-history Transformer → state fusion
→ DeepSets fleet summary → hierarchical routing → frvcpy nonlinear charging.

Training used 80 synthetic teachers, interleaved multi-open-route imitation
(~37–40% multi-open steps), 300 BC + 30 RL updates, seeds 42/43/44. Checkpoints
are versioned `*_c2.pt` and do not overwrite Cycle 1.

Pooled over 60 official instance-seed evaluations (all 60/60 feasible):

- Equal-budget H0: 22.062702 h
- Strong H0 + local search: 20.141634 h
- H1: 21.537649 h
- H2 (history, no DeepSets): 21.048156 h
- H3 (DeepSets, no history): 21.183247 h
- H4 (history + DeepSets): 20.958189 h

H4 improves on equal-budget H0 by 5.01% and slightly on H2 by 0.43% mean
objective, but wins only 28.3% of paired H4-vs-H2 comparisons. H4 remains
4.05% worse than H0_strong and beats it on only 16.7% of pairs. DeepSets
diagnostics show permutation invariance and nonzero gradients; the coordination
signal is real but weak for this static offline benchmark.

## Scale

Zero-shot n=10→n=20 (20 official instances, seed-42 c2 checkpoints, 5 samples):
all methods 20/20 feasible. Means: H0 36.690980; H0_strong 34.420101;
H1/H3 34.658276; H2 34.594263; H4 34.579682. H4 is the best neural mean and
within 0.16 h of H0_strong.

Zero-shot n=10→n=40 (5 instances, 3 samples): all 5/5 feasible. Means:
H0 63.747925; H0_strong 60.168459; H1 62.057388; H2 61.469290; H3 62.218225;
H4 61.921243. H2 is the best neural mean; H4 does not beat H2 or H0_strong.

## Learned charging

Cycle 2 direct CHARGE uses physical Wh candidates and time-aware masks with
exact nonlinear curves. Feasibility remains 4/20 on official n=10; hybrid H4
stays 20/20. Post-repair feasibility is due to frvcpy, not learned heads.

## Cycle 1 contrast

Under sequential IL and smaller data, H2 clearly beat H4. Cycle 2’s interleaved
training and richer tokens close that gap and slightly reverse the mean ranking,
without establishing DeepSets superiority or neural dominance over strong H0.

## Reproduction

Commands and per-seed tables are in `results/notes.md`. CSVs, learning curves
(`learning_curves_c2.png`), and checkpoints live under `results/` and
`checkpoints/`.
