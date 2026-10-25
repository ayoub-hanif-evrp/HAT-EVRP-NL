# Repository cleanup audit

Reviewed from commit `1466b2c`. Goal: publication-ready layout without changing the research architecture or fabricating results.

## Inventory summary

| Category | Count (approx.) | Disposition |
|---|---:|---|
| Core source (`*.py` root) | 8 | Retained at repository root |
| Montoya XML | 120 | Retained read-only under `dataset/` |
| Checkpoints `*.pt` | 59 | Retained; roles in `checkpoints/manifests/checkpoint_manifest.csv` |
| Experiment drivers | 8 | Moved to `scripts/experiments/` |
| Reporting scripts | 2+ | Under `scripts/reporting/` |
| Results CSVs / JSON | ~160+ | Layered under `results/raw/{phase}/` |
| Historical reports | ~12 | Archived under `results/archive/reports/` |
| Legacy PNGs / SVG | ~12 | Archived under `results/archive/figures_legacy/` |
| Final PNG figures | ≤10 | `results/figures/` (PNG only) |
| Final MD tables | 9 | `results/tables/` |

Machine-readable move list: [`CLEANUP_MOVE_MANIFEST.csv`](CLEANUP_MOVE_MANIFEST.csv) (SHA-256 + rationale per file).

## Classification legend

- **A** Core source code
- **B** Dataset / literature reference
- **C** Current reproducibility requirement
- **D** Current official numerical result
- **E** Historical but scientifically valuable
- **F** Derived output that can be regenerated
- **G** Confirmed obsolete / disposable (delete only after proof)

## Layout after cleanup

```text
scripts/
  experiments/     # run_* experiment drivers
  reporting/       # build_results.py, summarize_perf.py
results/
  README.md        # definitive index
  figures/         # publication PNGs only
  tables/          # publication Markdown tables
  raw/             # authoritative CSVs/JSON by research phase
  archive/         # older reports and legacy figures
  examples/        # small illustrative artifacts
checkpoints/
  manifests/       # checkpoint roles + hashes
docs/
  CLEANUP_AUDIT.md
  CLEANUP_MOVE_MANIFEST.csv
```

## Path centralization

`reporting_paths.py` holds `ROOT`, `DATASET`, `CHECKPOINTS`, `RESULTS`, `FIGURES`, `TABLES`, `RAW`, `ARCHIVE`, and authoritative CSV aliases used by the reporting pipeline.

## Deletion manifest (category G only)

Default policy is **archive, not delete**. No scientifically necessary XML, BKS,
Cycle-3 primary checkpoints, or authoritative CSVs were destroyed.

| Path | Action | Justification |
|---|---|---|
| `results/figures/fig_architecture.svg` | Archived → `results/archive/figures_legacy/` | Replaced by `fig01_architecture.png` |
| Legacy `results/figures/fig_*.png` (pre-cleanup) | Archived → `results/archive/figures_legacy/` | Superseded by numbered publication PNGs |
| `HAT_EVRP_NL_AGENT_PROMPT.md` | Deleted from `docs/archive/` | Old instruction file, not used for reproduction |
| Category G hard deletes | **None** | Ambiguous files archived under `results/raw/` or `results/archive/` |

No checkpoint `.pt` files were deleted (59 retained; roles in
`checkpoints/manifests/checkpoint_manifest.csv`).

## Nonlinear charging visibility

Older recovery CSVs stored charge **visit counts** only. Charging hours cannot be recovered from stop counts. The reporting pipeline re-executes the recovered ordinary λ=0 + hybrid protocol and writes:

`results/raw/regression_recovery/RECOVERY_n40_objective_decomposition.csv`

with travel / service / nonlinear charging / full / lit hours, verifying identity and (where possible) matching against the recovered pipeline objectives.

## Equal-budget honesty

`stage_fair_budget_compare()` is documented and reported as **matched starts, unequal realized compute**. Reports must not claim equal oracle budgets unless budgets are enforced.

## Regenerating presentation outputs

```bash
python scripts/reporting/build_results.py --all
```

## Validation checklist

- [ ] Unit tests pass (`python -m unittest discover -s tests -v`)
- [ ] Reporting tests pass
- [ ] Final figures are PNG; no SVG remains in `results/figures/`
- [ ] Tables are Markdown under `results/tables/`
- [ ] Dataset XML hashes recorded under `results/raw/references/`
- [ ] Historical n40 means intact in `raw/regression_recovery/` and `raw/cycle3/`
- [ ] README / paper no longer cite ~43% as the current BKS gap
