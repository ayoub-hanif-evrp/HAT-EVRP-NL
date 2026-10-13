# HAT-EVRP-NL — Targeted improvement report (2026-10-13)

**Base:** commit `66ade7e`  
**Architecture:** unchanged (Transformer → fusion → DeepSets → hierarchical decoder → frvcpy → Python validation).  
**No Java. No Cycle-3 checkpoint overwrite. No official-instance training.**

## Problems confirmed

| ID | Issue | Status |
|---|---|---|
| A | H2 λ=2 vs H4 λ=4 confounds DeepSets | **Fixed by matched-λ table** |
| B | Strong LS may starve late operators | Instrumented; on n=10 budget=4000 late ops *are* used; balanced schedule = no synth gain |
| C | Route head not cost-aware | Implemented `route_lambda`; **rejected** (no synth gain) |
| D | DeepSets aggregates closed routes | Documented; **no retrain** this cycle (matched λ already shows no reliable H4 win) |
| E | Ambiguous imitation interleaving | **Deferred** (inference first) |
| F | 20 instances / 5 layouts | **Reported** layout-clustered CIs |

Transparency: prior IMPROVEMENT cycle used official instances while selecting λ (documented). This cycle’s architectural conclusions use **matched** λ grids on official data; route_λ / balanced LS selection used synthetic pools only.

---

## Code changes

1. `solver_ls.py` — local search with budget checks before every oracle call (incl. both halves of two-route moves), per-operator diagnostics, schedules `basic` / `strong` / `strong_balanced`.
2. `train.decode` — optional `route_lambda` (top-k route shortlist + NEW_ROUTE potentials); `cost_lambda` unchanged.
3. `evaluate.py` — `--local-operators`, `--route-lambda`; refinement returns diagnostics.
4. Helpers remain for dual objectives (full vs literature).

---

## Phase 1 — Matched λ architectural ablation (primary)

Protocol: existing c3 checkpoints; best-of-10 + **identical** `operators=strong` (3 passes / 4000 FRVCP evals); seeds 42–44; all 20 n=10 instances.

File: `TARGETED_matched_lambda_strong_n10.csv`

| Method | λ | Mean full h | Mean lit BKS gap | Oracle/inst |
|---|---:|---:|---:|---:|
| H2 | 0 | 19.495 | 2.44% | 1032 |
| H2 | 2 | 19.487 | 2.40% | 1029 |
| **H2** | **4** | **19.335** | **0.60%** | 1047 |
| H4 | 0 | 19.458 | 2.21% | 1037 |
| H4 | 2 | 19.402 | 1.56% | 1039 |
| H4 | 4 | 19.407 | 1.47% | 1048 |

### Matched H4 − H2 (seed-averaged instances; layout-clustered CI)

File: `TARGETED_matched_H4_minus_H2.csv` · **5 unique customer layouts**

| λ | Mean Δ (h) | W/T/L | Layout mean | Layout 95% CI |
|---:|---:|---:|---:|---|
| 0 | −0.037 | 8/11/1 | −0.037 | [−0.087, +0.013] |
| 2 | −0.085 | 7/6/7 | −0.085 | [−0.196, +0.027] |
| 4 | **+0.071** | 7/7/6 | +0.071 | [−0.105, +0.248] |

**Conclusion:** Under matched decoder/search settings, **H4 does not reliably outperform H2**. Every layout-level CI includes zero. At λ=4, H2 is better on mean. The earlier “H4 1.47% gap” result mixed architecture with a more favorable λ for H4.

**Best verified configuration this cycle (not an architectural claim):** H2 + λ=4 + strong LS → **19.335 h** full, **~0.60%** mean lit BKS gap on n=10.

Separately, individually tuned bests (exploratory; not for DeepSets claims): H2 λ=4 and H4 λ=2/4 remain close.

---

## Phase 2 — Balanced local search

Synth pool: 40 feasible instances.  
`strong` vs `strong_balanced`: **40/40 ties**, mean Δ = 0.  
Mean evaluations ≈973 both.

On official n=10 with budget 4000, sequential strong already exercises swap / 2-opt* / eliminate (aggregate oracle share: two_opt_star largest).  

**Retain** `strong` as default; keep `strong_balanced` available for diagnostics. No quality gain → do not replace strong.

---

## Phase 3 — Route-level cost awareness

Synth (20 feasible): `route_lambda ∈ {0, 0.5, 1, 2}` with H2, cost_λ=0, strong LS → **identical** mean 15.070 h and oracle counts.

**Reject** enabling `route_lambda` by default. Code retained behind `route_lambda=0` default.

---

## Phase 4 / 5 — DeepSets open-only & imitation

**Not executed.** Matched-λ ablation already fails to support a DeepSets win; open-only would require new training. Imitation fixes deferred.

---

## Statistical note

Official n=10 = **5 geometries × 4 station configurations**. Instance-level n=20 overstates geometric independence. Layout-clustered intervals are the appropriate uncertainty for architecture claims.

---

## Retained vs rejected

| Change | Verdict |
|---|---|
| Matched-λ reporting | **Retain** (required for science) |
| Cost-aware customer λ (optional) | Retain as option; pick λ on synth going forward |
| Strong LS + diagnostics | **Retain** |
| Strong_balanced | Available; not better on synth |
| Route_lambda | Implemented; **disabled** (no synth gain) |
| Open-only DeepSets retrain | **Not run** |
| Symmetry-aware imitation | **Not run** |

---

## Hard stop summary

- Best verified full objective (n=10, this cycle): **H2 λ=4 + strong LS = 19.335 h** (lit gap **~0.60%**).
- **H4 does not improve over H2 under matched λ ∈ {0,2,4}** (layout CIs include 0; at λ=4 H2 wins on mean).
- Prior H4 “best gap” was **confounded** by unequal λ.
