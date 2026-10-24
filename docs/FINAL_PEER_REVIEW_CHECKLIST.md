# Peer-review checklist

Reviewed locked evaluation: `7671f6411a37666212656121d02a7c3af41ce1f7`. Charging decomposition: `91d7e6121de2bf03a555db6fec21a265d36d1151`. The statistics, holdout, manuscript, and provenance files in this review are later working-tree edits. No new BC or RL training was run.

| Requirement | Evidence | Status |
|---|---|---|
| Five seed differences, t interval, sign test, Wilcoxon, Holm | `results/tables/final/statistical_sensitivity.csv`, `docs/FINAL_STATISTICAL_ROBUSTNESS.md` | PASS |
| Claims weakened where the sign test is 0.0625 | `paper_draft.md` abstract and Section 8 | PASS |
| Four deployment options with time, executions, requests, hits, 2019 gap | `results/tables/final/deployment_options.csv`, `docs/QUALITY_COMPUTE_TRADEOFF.md` | PASS |
| Sample mean distinguished from best-of-four | Abstract, Section 8, `results/tables/final/sample_expectation_vs_selection.csv` | PASS |
| One frozen shift, protocol fixed first, no reselection | `results/raw/shift_holdout_v1/frozen_protocol.json`, completed in 39 s; sampled-mean delta +0.063 h | PASS |
| Negative holdout retained | `results/tables/final/shift_holdout_summary.csv` | PASS |
| Three charging layers and the 9.971 h to 10.117 h case | `paper_draft.md` Sections 3 and 10 | PASS |
| No claim to have invented `frvcpy` or discovered nonlinear charging | `docs/NOVELTY_AND_LITERATURE_POSITIONING.md` | PASS |
| Reference file audited against Table 13 | All 60 CSV values match Froger et al. (2022) Table 13; 18 of 20 n=10 values also match 2019 Table 9 | PASS |
| Related-work table with non-identical variants | `paper_draft.md` Section 2 | PASS |
| Reproducible commands and library versions | `results/tables/final/reproducibility_manifest.json`, `requirements.txt` | PASS |
| Existing test suite, 110 tests | `results/raw/final_test_transcript.txt` | PASS |
| Ruin-and-repair kept out of the proposed method | Section 8 and Section 12 | PASS |
| Five-seed linear intervention on the final weights | Not run; Cycle-3 evidence remains labeled as Cycle-3 | NOT COMPLETED |
| Current best-known gaps against Wang et al. (2025) or Froger et al. (2022) | Instance-level later solutions were not verified into a new gap file | NOT COMPLETED |
| n=80 four-sample evaluation | Not run | NOT COMPLETED |

The two NOT COMPLETED literature-gap and n=80 items are limitations, not missing claims. The manuscript does not treat the 2019 file as the current best-known set, and it says n=80 samples were not run.
