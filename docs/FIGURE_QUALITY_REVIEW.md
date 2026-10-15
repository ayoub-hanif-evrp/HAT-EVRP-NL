# Figure Quality Review

Visual inspection of every main-paper PNG after the scientific-results rebuild.
Inspection performed at full size and at approximate single-column width.

| Fig | Defects found | Fix applied |
|---|---|---|
| 01 | Architecture generally clear | Kept; neural vs frvcpy separation retained |
| 02–09 | Minor; previously audited | Regenerated with shared style |
| **10** | Twin-axis bars; long overlapping x-labels; legends crowding | **Three panels** (visits / energy / charge h); short H2/H4 labels; no twin axis |
| **11** | Legend clipped at top; long y-labels | Legend above axes; short labels; segment values inside bars; totals outside |
| 12 | Acceptable after prior fix | Kept (labels past error-bar tips) |
| **13** | Treated 60 instance×seed pairs as if independent; SE overstated independence | **Layout-mean deltas** (n=20); SE over layouts; explicit descriptive note |
| **14** | Only n10 fair-LS ~5%; hid stronger hybrid ~0.03% and n40 ~17% | **Three protocol panels** with non-mixing disclaimer |
| 15 | n80 warning present | Kept |
| **16** | Connecting lines implied sequential trajectory | **Scatter only** (no lines); categorical note |
| 17 | Legend outside (prior fix) | Kept; unequal-budget note retained |
| **18** | Overcrowded multi-architecture overlay | **2×2 architecture panels** |
| 19 | Slope plot OK | Kept; construction-only caveat reinforced in caption |
| **20** | Risk of reading noise as learning | Caption: noisy traces ≠ consistent learning; link to Fig 19 |
| **21** | Diagnostic only | **Moved to supplementary**; not a main claim figure |

## Acceptance checklist (post-fix)

- [x] No twin-axis confusion on Fig 10  
- [x] No clipped legend on Fig 11  
- [x] Layout-aware uncertainty on Fig 13  
- [x] Protocol-separated BKS on Fig 14  
- [x] No sequential lines on Fig 16  
- [x] Legible BC panels on Fig 18  
- [x] Reward caveat on Fig 20  
- [x] Fig 21 not in `results/figures/`  
- [x] Raw CSV identities unchanged  
- [x] Contact sheet regenerated  

## Note on automated tests

Tests check inventory, openability, raw-count identities, and paper embeds.
They **do not** prove visual non-overlap; human inspection of this review + contact sheet is required.
