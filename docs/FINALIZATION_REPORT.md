# Finalization Report — HAT-EVRP-NL

Baseline reviewed commit: `966010a`. This report documents the finalization work
executed on that HEAD.

## A. Implementation status

| Fix | Files | What was wrong | Correction | Test | Result |
|---|---|---|---|---|---|
| Strict LS deadline | `solver_ls.py`, `evaluate.py` | Soft budget shrink; LS not deadline-aware | `local_search(..., deadline=, clock=)`; stop before new evals; record overtime | `DeadlineLocalSearchTests` | PASS |
| Paired stats | `scripts/analysis/paired_stats.py` | Averaged mismatched seed/feasibility | Pair on instance×seed; duplicates/unmatched; feasibility separate; bootstrap CI | `PairedStatsTests` | PASS |
| Validation | `charging.py`, `tests/test_finalization.py` | Unknown node IndexError; incomplete fixtures | Bound-check successor nodes; deterministic fixtures | `ValidationFixtureTests` | PASS |
| LS attribution | `solver_ls.py` | (already fixed) + stronger test | Credit last improving operator | `OperatorAttributionControlledTests` | PASS |
| Artifact protection | `artifact_io.py`, `train.py` | Silent overwrite of ckpts/CSVs | `safe_save_checkpoint` / `safe_write_csv`; protected `*_c3.pt` | `ArtifactProtectionTests` | PASS |
| Fingerprint reanalysis | `scripts/experiments/recompute_route_change_fingerprints.py` | Vehicle-order sensitive rates | Normalize via `route_sequence_fingerprint` in versioned dir | `FingerprintRecomputeTests` + executed script | PASS |
| CI / discovery | `tests/__init__.py`, `.github/workflows/ci-fast.yml` | Incomplete CI | Full `unittest discover` + compileall + sanity | Local discover | PASS; GitHub Actions **NOT VERIFIED** remotely |

## B. Test evidence

Commands executed:

```text
python -m compileall -q .
python -m unittest discover -s tests -p 'test_*.py' -q
python evaluate.py --mode sanity --data dataset
python scripts/experiments/recompute_route_change_fingerprints.py
python scripts/experiments/run_lightweight_architecture_compare.py
```

- Unit tests: **73 ran, 0 failed** (ResourceWarnings only).
- Sanity: passed (`tc0c10s2cf1`).
- Environment: local Windows; Python via project venv/system; PyTorch+frvcpy available.
- GitHub Actions run on `main` after push: **NOT VERIFIED** in this session (workflow updated locally; requires push + Actions check).

## C. Experiment evidence

### Executed

1. **RQ3 fingerprint reanalysis**  
   Dir: `results/raw/nonlinear_charging_reanalysis/fingerprint_invariant/966010a2_*`  
   Invariant route-change rates (unchanged vs prior at n=40): SECANT 30%, EARLY 25%.

2. **Lightweight P0 n=10 architecture compare** (existing ckpts, seed 42)  
   Dir: `results/raw/architecture_compare/lightweight_p0_n10/966010a2_*`  

   | method | feasible | mean Z_full |
   |---|---:|---:|
   | H1 | 20/20 | 22.499 |
   | H2 | 20/20 | 21.844 |
   | H3 | 20/20 | 22.668 |
   | H4 | 20/20 | 21.844 |
   | H4_open* | 20/20 | 21.800 |

   \*H4_open uses a **different** training tag (`perf_v2mt_s`), not Cycle-3 matched.

3. **Paired fair-LS (seeds 42–44 pooled, instance×seed pairs)**  

   | pair | mean Δ (B−A) | n | W/T/L | bootstrap 95% CI |
   |---|---:|---:|---|---|
   | H2−H1 | −0.180 | 60 | 17/33/10 | [−0.344, −0.019] |
   | H4−H2 | +0.074 | 60 | 9/41/10 | [−0.036, +0.191] |

### Not executed (manifest: `results/raw/experiment_manifest.yaml`)

- H2-prefix history training (RQ1 causal ablation) — encoder path not wired.
- Matched multi-seed H4-mean / DeepSets sweep (RQ2).
- 2×2 oracle×feature factorial (RQ3).
- RL under hybrid P2 (RQ4).
- 5-seed confirmatory fair-LS n10/n20/n40.

## D. Final scientific assessment

1. **Does H2 reliably outperform H1?**  
   **Yes, modestly** under Cycle-3 fair-LS n=10 (paired bootstrap CI for H2−H1 excludes 0). P0 construction also shows H2 < H1 mean.

2. **Does H4 reliably outperform H2?**  
   **No.** Paired fair-LS mean Δ is slightly positive (H4 worse); CI includes 0. P0 seed-42 H2=H4 exactly.

3. **Does DeepSets help larger fleets?**  
   **Not established.** n40 hybrid mean favors H4 but layout CI includes 0 (prior audit). H4_open looks competitive on P0 but is not a matched Cycle-3 checkpoint.

4. **Nonlinear charging impact?**  
   **Strong.** TRUE-NL 20/20; SECANT/EARLY ~11/20 fixed-route n40; e2e EARLY 9/20. Fingerprint-invariant route changes remain material (25–75% by size/model).

5. **Does RL improve the final hybrid solver?**  
   **Not shown.** Controlled RQ4 is construction-only (H4 3/3, H2 1/3). Hybrid RL eval deferred.

6. **Competitiveness vs BKS?**  
   Protocol-dependent: ~5% lit gap (n10 fair-LS H2) vs ~0.03% (expensive hybrid_insert) vs ~17% (n40 recovered). Not SOTA as an unqualified claim.

7. **Learned model vs classical optimizer?**  
   History helps vs state-only under matched LS. Hybrid near-BKS gains are dominated by search/`frvcpy`, not architecture alone.

8. **Limitations before publication**  
   Prefix-history ablation; matched DeepSets pooling multi-seed; equal wall-clock P3 with deadline (code ready, full table not run); hybrid RL; remote CI verification.

## E. Repository readiness checklist

| Item | Status |
|---|---|
| Training logging | PASS |
| Route validation | PASS |
| Charging validation | PASS |
| Objective consistency | PASS |
| Solver-execution accounting | PASS |
| Strict wall-clock benchmarking | PASS (code+unit); full P3 table NOT VERIFIED |
| Route fingerprints | PASS |
| Operator attribution | PASS |
| Deterministic tests | PASS |
| Complete test suite | PASS (73/73) |
| CI workflow | PASS (file); GitHub run NOT VERIFIED |
| Checkpoint protection | PASS |
| Result reproducibility | PASS (versioned run dirs) |
| Statistical correctness | PASS (script+tests) |
| RQ1 evidence | PASS (modest; prefix ablation NOT VERIFIED) |
| RQ2 evidence | PASS (negative/inconclusive documented) |
| RQ3 evidence | PASS |
| RQ4 evidence | PASS (construction-only); hybrid NOT VERIFIED |
| Competitive benchmarks | PASS (protocol-separated; not SOTA) |
| Manuscript consistency | PASS (conservative; update claims if needed) |

## Definition of done — honest status

1. Remaining **implementation** defects for Stage A: **done and tested**.  
2. Protocol reproducible: **yes** (manifest + scripts).  
3. Confirmatory multi-seed training ablations: **not fully executed** (disclosed).  
4. Figures/tables agree with prior raw data: **yes**; new lightweight CSVs versioned.  
5. Conclusions do not exceed evidence: **yes**.  
6. Unresolved items: **explicitly disclosed** above.
