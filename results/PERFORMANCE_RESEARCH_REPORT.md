# HAT-EVRP-NL Performance Research Report

Evidence-driven continuous improvement after Cycle-3 / matched-λ work (`4c1a5f5`).
Architecture preserved: Causal EV History Transformer → Dynamic State Fusion → DeepSets →
hierarchical pointer decoder → `frvcpy` → Python validation. No Java. Cycle-3 checkpoints
(`*_c3.pt`) were never overwritten.

## Step 0 — Baseline

- Sanity (`tc0c10s2cf1`) passed; Cycle-3 H2/H4 checkpoints load.
- Prior exploratory best (official n=10, matched search): **H2 λ=4 + strong LS ≈ 19.335 h full / ≈ 0.60% lit BKS gap**.

## Step 1 — Synthetic generator v2 — RETAIN

Defect: geometry / tech / placement / density were coupled via `seed % k`.

Fix (`data.py`):

- Seed only initializes RNG.
- Independent `SyntheticConfig`: geometry, station placement, density, tech pool, cluster σ.
- P-median uses **region-sampled candidates**, not customer copies.
- `generator_version='v1'` retained for ablation.

Verification (`PERF_generator_coverage.csv`, 240 seeds @ 500000+):

| Check | Result |
|---|---|
| Reproducibility | 240/240 |
| Zero-detour stations | 0 |
| Geometry↔tech decoupled | yes |
| Explicit factor combos | verified |
| Geometry mix | random 91 / clustered 75 / mixed 74 |
| Placement | uniform 130 / pmedian-region 110 |
| Tech tags | f,n,s,fn,fs,ns,fns all present |

## Step 2 — Stronger teacher — RETAIN

Paired compare on 16 v2 instances (`PERF_teacher_compare.csv`):

| Outcome | Count |
|---|---:|
| Strong better | 13 |
| Basic better | 0 |
| Tie | 1 |
| Both infeasible | 2 |

Strong teacher (8 restarts, 3 strong-LS passes, budget 2500) dominates within modest extra oracle cost.

## Step 3 — Multi-valid-action imitation — OPTIONAL / NOT PROMOTED

Implemented optional `multi_target` loss: \(-\log\sum\) hierarchical probs over expert-consistent
(route, customer/close) pairs (`train.py`).

Matched pilot (v2, strong teacher, 60/20, 100 steps, seed 42):

| Loss | Best val mean full h |
|---|---:|
| Single-target (`perf_v2st`) | **≈16.52** |
| Multi-target (`perf_v2mt`) | ≈17.54 |

**Decision:** keep multi-target as selectable code path; promote **single-target** for pilots.

## Step 4 — Open-only DeepSets (H4_open) — NO CLEAR GAIN

`PolicyConfig.deepsets_open_only` aggregates only open routes. New `H4_open` variant trained
from scratch. Diagnostics: perm-error ≈ 0, nonzero φ gradients, multi-open fraction ≈ 0.46.

On matched validation and official tests, H4_open ≈ H4 ≈ H2 under identical training; **no
reliable coordination benefit**.

## Step 5 — Controlled training

Pilots (new seed ranges 300000+/400000+, never overwrite c3):

| Checkpoint | Notes |
|---|---|
| `H2/H4/H4_open_*_perf_v2st.pt` | v2 + strong teacher + single-target (promoted) |
| `*_perf_v2mt.pt` / `*_perf_v2mt_s.pt` | multi-target / scaled; not promoted |
| `*_c3.pt` | previous best for official n=10 |

Synthetic holdout (24 seeds @ 520000+, λ=4, strong LS): all methods ≈66.7% feasible;
feasible mean objs within ~0.08 h of each other (c3 slightly best). Hard v2 instances remain
challenging for all policies.

## Step 6 — Ruin-and-repair — RETAIN (search)

Bounded destroy/repair with FRVCP deltas (`solver_ls.ruin_and_repair`). Diagnostics now
separate evaluations / feasible candidates / provisional updates / accepted moves.

Synthetic ablation (matched budget; `PERF_search_ablation.csv`):

| Search | Mean full h (feas) | Ruin W/T/L |
|---|---:|---|
| Strong LS only | 17.738 | — |
| Strong LS + ruin | **17.617** | **2 / 6 / 0** |

Route-pool assembler implemented (`assemble_route_pool`) for future portfolio labeling;
not promoted as an architectural claim.

## Step 7–8 — Official benchmarks

Protocol: 8 (n=10) / 4 (n=20) constructions, λ=4, then one-shot strong LS (3×4000).
Literature gap uses service-excluded objective vs Froger Table 13 BKS.
Layouts: tc0/tc1/tc2 (five geometries × charging variants on n=10).

### n=10 (seeds 42–43 pooled where available)

| Method | Feas% | Mean full h | Mean lit gap% | Median lit gap% |
|---|---:|---:|---:|---:|
| **H2_c3** | 100 | **19.366** | **0.81** | 0.01 |
| H4_c3 | 100 | 19.447 | 1.39 | 0.01 |
| H2_new (v2st) | 100 | 19.791 | 5.15 | 0.02 |
| H4_new | 100 | 19.650 | 3.72 | 0.02 |
| H4_open_new | 100 | 19.853 | 5.45 | 0.53 |

Layout-clustered lit gaps (pooled):

| Method | tc0 | tc1 | tc2 |
|---|---:|---:|---:|
| H2_c3 | 1.35 | 0.73 | 0.50 |
| H2_new | 0.12 | **8.05** | 1.50 |
| H4_c3 | 1.36 | 1.73 | 0.40 |
| H4_new | 0.01 | 5.95 | 0.73 |

New v2-trained policies improve some tc0 instances but regress badly on tc1 → **do not replace c3 on n=10**.

### n=10 + ruin (H2_c3 seed42 only; search add-on)

`PERF_official_n10_H2c3_ruin.csv`: mean full **19.316 h**, mean lit gap **≈0.37%**.
This beats the prior 19.335 / 0.60% exploratory figure via **search**, not architecture.

### n=20 (seed 42, all 20 instances)

| Method | Feas% | Mean full h | Mean lit gap% |
|---|---:|---:|---:|
| H2_c3 | 100 | 31.306 | 11.24 |
| **H2_new** | 100 | **30.953** | **8.84** |
| **H4_c3** | 100 | **30.593** | **7.03** |
| H4_new | 100 | 30.991 | 9.14 |
| H4_open_new | 100 | 30.953 | 8.84 |

On n=20, H2_new beats H2_c3, but **H4_c3 remains strongest** under matched λ/search.
n=40 / n=80 not fully executed in this cycle (resource/time).

## Step 9 — Quality targets vs outcomes

| Target | Status |
|---|---|
| 100% feasible on evaluated official cases | Met (n=10, n=20) |
| Better synth val than Cycle-3 | Not met (tie / slight c3 edge) |
| Better matched-budget quality | Ruin helps; new n=10 policy does not |
| Match/beat ~0.60% n=10 lit gap | **Search:** ≈0.37% with H2_c3+ruin; **new policy:** worse mean |
| Improve n=20/40 | n=20: H2_new improves vs H2_c3; H4_c3 still best |
| H4 > H2 under identical settings | **Not supported** (n=10); H4_c3 wins n=20 but vs H2_c3 only |

Three successive training-axis experiments (multi-target, open-only DeepSets, v2+strong
pilot) did not beat Cycle-3 on official n=10 validation of architectural gains. Plateau
diagnosed as **domain shift / under-capacity pilots** on Montoya layouts (esp. tc1), while
**search (ruin)** and **stronger teachers** are positive, measurable components.

## What worked / failed

**Worked**

1. Generator decoupling (coverage + zero-detour fix).
2. Stronger FRVCP teachers (13–0–1).
3. Bounded ruin-and-repair (synth 2/6/0; official H2_c3 gap 0.60% → ≈0.37%).
4. Instrumented LS counters; pandas declared in `requirements.txt`.

**Failed / not promoted**

1. Multi-target imitation (worse matched val).
2. Open-only DeepSets (no H4 architectural win).
3. Replacing Cycle-3 n=10 policies with short v2 pilots (tc1 regression).
4. Claiming H4 > H2 under matched settings.

## Best validated configurations (this cycle)

1. **Practical n=10 quality:** H2 Cycle-3, `cost_lambda=4`, strong LS + bounded ruin  
   → ≈19.316 h full / ≈0.37% mean lit BKS gap (seed 42).
2. **Matched architecture compare:** H2_c3 vs H4_c3, λ=4, strong LS (no ruin confound)  
   → H2_c3 preferred on n=10; H4_c3 preferred on n=20.
3. **Training stack for further research:** v2 generator + strong teacher + single-target BC  
   (multi-target / open-only kept as ablation flags).

## Reproducibility

```bash
python evaluate.py sanity --instance tc0c10s2cf1
python results/run_performance_research.py --stage coverage
python results/run_performance_research.py --stage teacher --teacher-n 16
python train.py --model H2 --size 10 --steps 100 --train-instances 60 --val-instances 20 \
  --seed 42 --version perf_v2st --teacher strong --generator v2
python results/run_performance_research.py --stage synthval --synth-n 24 --version perf_v2st
python results/run_performance_research.py --stage search --search-n 12
python results/run_performance_research.py --stage official --version perf_v2st --official-seeds 42,43
python results/run_performance_research.py --stage official20 --version perf_v2st
python results/summarize_perf.py
```

Artifacts: `PERF_*.csv`, `PERF_summary.json`, checkpoints `*_perf_v2st.pt` / `*_perf_v2mt*.pt`.

## Limits

- Pilot training (60–120 instances, ≤180 BC steps) — not full 800–1500 scale.
- Official n=40/80 not completed this cycle.
- n=10 “best gap” with ruin is still exploratory (hyperparameters inspected on the same benchmark).
- Hardware: CPU PyTorch training; wall-clock dominated by FRVCP oracle calls.
