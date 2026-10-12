# History-Aware Transformer Agents with DeepSets Coordination for Electric Vehicle Routing with Nonlinear Charging

## Abstract

We study whether a shared causal visit-history Transformer combined with a
DeepSets fleet representation improves customer sequencing for the Montoya
E-VRP-NL when charging is optimized by an exact nonlinear fixed-route solver.
After correcting an unequal local-search protocol that previously overstated
neural effort, fair one-shot local search on existing checkpoints shows that
history-aware hybrid policies (H2/H4) improve on a strong classical
greedy+LS baseline on official n=10–40 instances, while DeepSets does not
reliably beat history alone. Direct learned charging remains largely
infeasible. Gaps to published Froger et al. (2021) BKS values remain large
(~41% for H2 on n=10), so these methods are not competitive with specialized
metaheuristics.

## 1. Introduction

Electric vehicle routing with nonlinear charging requires feasible battery and
duration decisions under piecewise-linear charging curves. We ask whether
history-aware multi-route neural construction, coordinated by DeepSets, helps
when charging is handled by `frvcpy`.

## 2. Problem

Montoya et al. (2017) E-VRP-NL (VRP-REP 2016-0020): unlimited homogeneous EVs,
exact customer coverage, \(T_{\max}=10\) h, partial nonlinear charging, no time
windows, cargo limits, or station queues. Objective: total route duration.

## 3. Method

HAT-EVRP-NL: shared causal Transformer over visit tokens (SOC, time, energy,
charging, node type) → state fusion → DeepSets fleet summary → hierarchical
EV/customer decoder → `frvcpy` charging → independent checker. Ablations H1–H4
isolate history and DeepSets.

## 4. Fair evaluation protocol

**Construction-only:** N sampled solutions without local search; report best.

**Fair local search (corrected):** N constructions without LS for every method;
keep the best; apply identical one-shot 2-opt/relocate LS with the same
evaluation budget. Prior Cycle-3 “eqls” runs applied LS to every neural sample
and are **not** equal-effort; they are retained only as legacy files.

**Fair wall-clock:** measure neural wall time per instance; spend the same
measured time on classical constructions, then one shared LS.

## 5. Results

### 5.1 Official n=10 (60 instance-seed evaluations)

Construction-only: H0_strong 20.142; H2 20.966; H4 20.975 (all 60/60 feasible).
History helps vs H1/H3; DeepSets does not beat H2; H0_strong remains best.

Fair one-shot LS: H0_strong 20.142 (528 oracle/inst); H2 19.813 (747);
H4 19.886 (753). H2 and H4 beat H0_strong on 60% of pairs. H4 does not beat H2
(mean delta +0.074 h; win rate 18%).

Fair wall-clock (seed 42): mean runtimes ≈0.44–0.49 s; H2 19.586; H4 19.840;
H0_strong 20.889.

### 5.2 Scale (seed 42, fair LS)

n=20: H0_strong 34.420; H2 32.332; H4 32.322 (20/20).
n=40: H0_strong 60.195; H2 57.173; H4 57.506 (20/20).

### 5.3 Literature BKS

Using Froger et al. (2021) Table 13 BKS, seed-42 fair-LS mean gaps on n=10 are
47.1% (H0_strong), 41.0% (H2), 42.7% (H4). Specialized ILS/LNS methods remain
substantially stronger.

### 5.4 Charging ablation

Direct hierarchical CHARGE policy: 4/20 feasible. Hybrid neural routing with
`frvcpy` charging: 20/20. Repair with `frvcpy` is not credited to learning.

## 6. Limitations

No Java runtime for the DIMACS checker in this environment (checker cloned;
sample MIP XML verified internally; H4 XML exported). CPU-only training.
DeepSets coordination not supported by paired tests. Large gaps to published
BKS. Incomplete 160/320 evaluation.

## 7. Conclusion

With a corrected equal-effort local-search protocol, history-aware Transformer
construction plus one shared nonlinear-aware LS improves on a strong classical
baseline on the tested Montoya sizes, but DeepSets adds no consistent gain, and
performance remains far from literature BKS. Learned nonlinear charging is not
yet reliable; exact charging optimization remains essential.
