# Novelty and literature positioning

Manuscript date for the citation check: 23 October 2026. This note does not claim that the search was an exhaustive systematic review.

## What is already known

Montoya et al. (2017) defined E-VRP-NL and the piecewise-linear station curves. Froger et al. (2019), doi:10.1016/j.cor.2018.12.013, gave formulations and an exact fixed-route charging algorithm, and showed that charging decisions change route quality. Kullman et al. (2021), doi:10.1287/ijoc.2020.1035, published `frvcpy`. Froger et al. (2022) and Wang et al. (2025), doi:10.1016/j.trc.2024.104932, report later E-VRP-NL solutions. Nafstad, Desaulniers, and Stålhane (2025), doi:10.1287/trsc.2024.0725, solve a time-window variant with heterogeneous recharging technologies by branch-price-and-cut. Kool et al. (2019) and Kwon et al. (2020) train neural constructors for TSP and CVRP. Accorsi, Lodi, and Vigo (2022) and Thyssens et al. (2023) state how neural routing should be timed and compared. None of those facts is a result of this repository.

## What this study adds

The policy learns route choice and customer order. Behavior cloning is followed by REINFORCE with a detached leave-one-out advantage. `frvcpy` solves charging conditional on the customer sequence. An independent checker recomputes the Montoya charging time. The experiments then measure, on frozen checkpoints, whether reinforcement learning changes greedy routes, the expected sampled route, travel time, and nonlinear charging time, and whether two stated linear planners remain feasible under the original curves.

The study does not claim to have discovered nonlinear charging, to have invented `frvcpy`, or to be the first neural method with nonlinear EV charging. The literature check above is not sufficient for a priority claim.

## What is not the same problem

Nafstad et al. (2025) include time windows. Wang et al. (2025) include a heterogeneous fleet and station waiting; their E-VRP-NL runs are a special case of a different algorithm, not a matched comparison with this constructor. Routing Arena evaluates neural solvers on standard routing benchmarks, not on Montoya E-VRP-NL. Numerical objectives from those papers are not placed next to the hours in `paper_draft.md`.

## Reference objectives

`results/literature_bks_froger2021.csv` keeps its filename and its historical source string. `results/literature_bks_provenance.json` records the 2019 article, the checked tables, the objective \(Z_{\mathrm{lit}}\), and the later papers that are not this file. Gaps in the manuscript are 2019 reference gaps.
