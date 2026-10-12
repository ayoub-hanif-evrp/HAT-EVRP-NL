# HAT-EVRP-NL — Final research summary

All reported solutions were checked by an independent forward simulator using
Montoya piecewise charging curves, service times, full initial battery, SOC
bounds, exact customer coverage, depot return, and the 10-hour route limit.
Published best-known solution tables with verified matching objectives were
not available locally; BKS gaps are reported as unavailable (`NA`). DIMACS
notes that Froger et al. (2021) report BKS values for this set, but we did not
obtain a verified digital BKS table for gap computation.

## Final architecture (unchanged)

Shared causal EV visit-history Transformer → dynamic EV state fusion →
DeepSets fleet summary
`g = ρ(∑ φ(EV_state_i + Transformer_hidden_i))` → hierarchical
route/customer/close decoder → `frvcpy` nonlinear charging optimization →
independent feasibility validation.

Visit tokens include SOC, route time, travel energy, charging amount/duration,
and node type. Customer-decision history is distinct from mutable optimized
charging plans. Training uses interleaved multi-open-route imitation so
DeepSets sees competing open EVs.

## Final training (Cycle 3 / `c3`)

- Synthetic teachers only (official Montoya held out for evaluation).
- 120 train / 12 validation feasible instances per seed; random/clustered/mixed
  customers; varied station layouts and technologies.
- 400 BC + 40 RL updates; best validation checkpoint retained.
- Seeds 42, 43, 44. Checkpoints: `checkpoints/*_c3.pt` (Cycle 1/2 preserved).
- Efficiency: cached node embeddings, route-history cache, batched customer
  scoring, singleton oracle reuse. Decision behavior unchanged.

## Main n=10 results (20 official instances × 3 seeds = 60)

### Equal construction samples (10 attempts; H0 without LS; H0_strong with LS)

| Method | Feasible | Mean duration (h) |
|---|---|---|
| H0 | 60/60 | 22.063 |
| H0_strong | 60/60 | 20.142 |
| H1 | 60/60 | 21.249 |
| H2 | 60/60 | 20.966 |
| H3 | 60/60 | 22.131 |
| H4 | 60/60 | 20.975 |

Paired H4−H2 mean delta +0.009 h; H4 better on only 33% of pairs.
Paired H4−H0_strong mean delta +0.833 h; H4 better on 12%.

History helps (H2/H4 beat H1/H3). DeepSets does not add a reliable gain over
history alone under equal samples. Strong classical LS still wins this budget.

### Equal local-search refinement (same LS budget applied to neural routes)

| Method | Feasible | Mean duration (h) |
|---|---|---|
| H0 | 60/60 | 22.063 |
| H0_strong | 60/60 | 20.142 |
| H1 | 60/60 | 19.644 |
| H2 | 60/60 | 19.601 |
| H3 | 60/60 | 19.702 |
| H4 | 60/60 | 19.597 |

Paired H4−H0_strong mean delta −0.545 h; H4 better on **76.7%** of pairs.
Paired H4−H2 mean delta −0.004 h; essentially tied (H4 better on 20%).

Under equal LS, neural construction provides better starting routes than
randomized greedy; H4 and H2 both beat H0_strong. DeepSets remains nearly
neutral relative to H2.

### Equal wall-clock (seed 42)

Giving H0 extra restarts to match neural wall time: H0 20.994; H0_strong
20.010; H2 20.792; H4 21.057. Classical methods regain the lead when runtime,
not construction count, is equalized.

## Scalability (seed-42 `c3` checkpoints, zero-shot)

| Setting | H0_strong | H2 | H4 | Feasible |
|---|---|---|---|---|
| n=20 equal samples | 34.420 | 33.708 | 33.820 | 20/20 |
| n=20 equal LS | 34.420 | 32.217 | 32.307 | 20/20 |
| n=40 equal LS | 60.195 | 57.173 | 57.506 | 20/20 |
| n=80 equal LS (3 inst.) | 95.506 | 88.151 | 88.151 | 3/3 |

Larger sizes were not fully enumerated for 160/320 customers under the CPU
budget; n=80 is an explicitly limited sample.

## DeepSets scientific evaluation

Diagnostics on H4: permutation error ≈ 0; nonzero φ gradients; nonzero change
in active-EV logits when other EV states change; training multi-open fraction
≈ 37–40%. Across Cycle 2 and Cycle 3, H4 vs H2 differences are small and
seed-sensitive. Conclusion: DeepSets is correctly implemented and active, but
provides little systematic objective benefit on this static offline benchmark
beyond history-aware routing.

## Nonlinear charging

Hybrid H4 (neural customers + `frvcpy`) remains 20/20 feasible.
Direct learned CHARGE (c3): **4/20** feasible despite physical Wh candidates,
time-aware masks, and multi-station reachability checks. Repair with `frvcpy`
restores 20/20; that gain is not attributed to the learned charging heads.
Direct CHARGE is retained as a negative ablation.

## Runtime

On n=10 equal samples (pooled), mean per-instance runtime ≈ 0.12 s (H0),
0.18 s (H0_strong), 0.46–0.67 s (neural). Caching reduced neural overhead
relative to Cycle 2 but neural inference + oracle calls remain slower than
classical construction.

## Strengths and limitations

Strengths: faithful Montoya formulation; verified feasibility; controlled
H1–H4 ablations; interleaved DeepSets training; equal-budget and equal-LS
comparisons; transparent negative CHARGE result.

Limitations: CPU-only training; BKS gaps unavailable; DeepSets gain not
established; wall-clock equalization still favors classical search; direct
charging remains unreliable; large-instance evaluation is partial.

## Artifacts

- Checkpoints: `checkpoints/*_c3.pt`
- Tables: `results/final_n10_equal_budget_c3.csv`,
  `results/final_n10_equal_ls_c3.csv`, `results/final_paired_c3.csv`
- Figures: `final_n10_comparison_c3.png`, `deepsets_paired_c3.png`,
  `scalability_c3.png`, `learning_curves_c3.png`
- Example verified solution:
  `results/example_solution_H4_c3_tc0c10s2cf1.json`
- Paper draft: `paper_draft.md`
- Experimental log: `results/notes.md`
