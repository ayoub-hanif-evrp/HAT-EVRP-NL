# HAT-EVRP-NL — Research Coding Agent Instructions

**Project:** History-Aware Multi-Agent Transformer for Electric Vehicle Routing with Nonlinear Charging  
**Working directory:** The current project directory. The benchmark is already in `./dataset/`.  
**Your role:** Research engineer implementing and evaluating a scientific method, **not** a software-product developer.

## 0. Mission and working style (read this first)

Our research question is:

> Can a shared, causal visit-history Transformer combined with a permutation-invariant global route/fleet representation improve customer allocation and sequencing for the Montoya E-VRP-NL, and can a hybrid nonlinear-charging optimizer produce stronger feasible solutions than a fully learned charging policy?

We want **real experiments**, reproducible numbers, and scientifically useful ablations. Follow the loop:

**inspect data → implement minimal working method → run → validate → measure → diagnose → improve one thing → rerun.**

Rules:

- **Do not build an application.** No web front end, API, database, microservices, Docker stack, elaborate package infrastructure, plugins, deployment, dashboards, or excessive abstractions.
- Keep code compact and readable: prefer **about 5–7 Python research files** and a short `requirements.txt`. Use functions/simple classes, not a hierarchy of interfaces. Do not create many placeholder files.
- **Do not produce a huge test suite.** Use a few targeted numerical/feasibility sanity checks integrated into the research scripts, and an independent checker for completed solutions when possible.
- **Do not stop after writing architecture or scaffolding.** Execute code, train at least a small model, and show actual results as the environment permits. Never invent performance or claim an unrun experiment succeeded.
- Do not over-engineer configuration: CLI arguments with sensible defaults are enough. Use PyTorch, NumPy, and common scientific Python packages.
- Use available GPU if present, otherwise CPU with smaller instances and batches. Record device, seeds, runtime, and code settings.
- Dataset files under `dataset/` are **read-only**. Do not rename, rewrite, or regenerate them. All models and outputs go to `results/` and `checkpoints/`.
- Keep a **brief** `results/notes.md` logging each meaningful experiment: setting, feasible rate, objective, time, interpretation, next action.
- If results are poor, **debug modeling/feasibility first**, not add complexity blindly.
- If a dependency is incompatible, diagnose it, try a lightweight fix, and record the issue. Do not spend the whole project rewriting an external optimizer.

## 1. Correct benchmark formulation — non-negotiable

Use **Montoya et al. (2017), Electric Vehicle Routing Problem with Nonlinear Charging Functions (E-VRP-NL)**, VRP-REP ID `2016-0020`, available in `dataset/`.

The original formulation is:

- One depot, customers with **service times**, charging stations with station-specific charging curves.
- **Unlimited number of homogeneous EVs**. Each new route begins at the depot with a **full battery** and returns to the depot.
- All customers must be served **exactly once**. Customers already committed to a route are unavailable to other routes.
- Each route must finish within the depot's common maximum **route duration `T_max`**.
- Travel has **time** and **energy** costs; energy and time calculations must reproduce the instance specification.
- Battery charge must remain between 0 and capacity `Q` throughout every route.
- Partial charging is allowed, using **concave piecewise-linear energy-vs-charging-time curves**; different station types may charge at different rates. Repeated visits to charging stations are allowed.
- No customer time windows, no vehicle cargo-capacity constraint, and **no charging-station queues/capacity** in the original Montoya E-VRP-NL. Do not accidentally implement Schneider's EVRPTW or Froger's NL-C as the primary task.
- Minimize **sum of route durations** (driving + customer service + charging time); **not** makespan, geographic distance only, or fleet count. Fleet count is a diagnostic, not an additional term in the published objective.

**Important detail:** Our EVs are *routes in a static offline optimization process*, not real-time vehicles making simultaneous physical moves. Use a construction index `k` and a **separate route elapsed time** for each EV. Do **not** implement an event-queue/time-window simulator unnecessarily.

### 1.1 Nonlinear charging implementation

For charging station `s`, parse all actual breakpoints `(charging_time, battery_level)` and build monotone piecewise-linear interpolation of the cumulative empty-to-energy charging time function `C_s(q)`. If entering with energy `q_in` and exiting with `q_out`, use:

`charging_time = C_s(q_out) - C_s(q_in)` for `q_in <= q_out <= Q`.

The implementation must obey the true benchmark units and rounding. **Inspect dataset fields** before assuming kWh vs Wh, km/h, decimals, or whether a library's returned route cost includes service time. No invented smooth sigmoid/linear replacement of the benchmark curve. Ensure that the interpolation/inverse accurately handles every segment and charging technology.

## 2. First action: investigate the actual local dataset

Before coding the solver:

1. List files recursively under `dataset/`; identify `.xml`, archives, best-known-solution files, documentation, and directory layout. Unpack archives only if needed, outside the original data.
2. Read a **10-customer** instance and examples from the 20/40/80/160/320 groups if present. Inspect depot/customer/station IDs, coordinate encoding, Euclidean distance rules, speed, consumption, service times, `T_max`, charging curves, and naming convention.
3. Confirm the benchmark's 120-instance, six-size-group structure (10, 20, 40, 80, 160, 320), but **report what is actually present** if the local folder differs.
4. Look for published solutions/BKS or a compatible independent checker. If BKS cannot be obtained, report absolute objectives and feasibility; do **not** invent gaps.
5. Print a compact data summary to console and `results/dataset_summary.csv`: instance name, N customers, stations, station types, battery, route limit, coordinate range, average service time.

**Deliverable / gate:** One small instance parses; all charging curves and travel/energy matrices are available; a hand-constructed route can be evaluated with correct time and SOC.

## 3. Minimal research code layout

Start with approximately this structure; merge files where sensible. Do not create empty modules just to follow a template.

```text
project/
├── dataset/                 # already exists; immutable
├── data.py                  # parser, instance representation, synthetic generator
├── charging.py              # exact charging curves, oracle adapter, verification
├── solver.py                # route construction, masks, heuristic, local search
├── model.py                 # node encoder, history transformer, DeepSets, policy
├── train.py                 # imitation / policy-gradient training
├── evaluate.py              # baselines, ablations, validation, CSV/plots
├── requirements.txt
├── checkpoints/             # created by training
└── results/                 # CSV, curves, examples, short research notes
```

If an additional file is absolutely needed, add it; otherwise keep the project simple. Use `argparse` and explicit random seeds. A few readable command-line calls should perform all experiments. Avoid a config framework.

## 4. First build a *verified* non-neural solver baseline

A model without a trustworthy objective/feasibility engine produces meaningless paper results. **Do this before ML.**

### 4.1 Fixed-route charging expert (preferred)

Investigate and try integrating **`frvcpy`**: https://github.com/e-VRO/frvcpy . It accepts a **fixed sequence of customers**, inserts charging stations, and optimizes charging amounts/charging time using a labeling algorithm for the fixed-route vehicle charging problem. It includes a translator for Montoya-like VRP-REP XML.

For a customer-only route `[depot, c1, c2, ..., depot]`, return:

- feasible/infeasible;
- exact/optimized duration under the supported model;
- full node sequence with any inserted stations, potentially multiple stations between customers;
- departure SOC / charging amount at each station.

**Verify its API, units, handling of service times, and interpretation of infeasible results using the installed version.** Do not assume it handles all benchmark details automatically. Add the appropriate `T_max` check if not enforced internally. Cache evaluated `(instance, ordered_customer_sequence)` results where possible. If the library cannot run, document the blocker and use a correct but potentially slower constrained charging routine for small instances, without falsely calling a heuristic exact.

### 4.2 Independent feasibility evaluator

Regardless of the oracle, implement a **small independent forward simulation** of a returned concrete route:

- starts/ends at depot;
- every customer belongs to one route and occurs once globally;
- energy deducted on each leg, no negative SOC or SOC above `Q`;
- charging time exactly from parsed piecewise curve;
- customer service times included exactly once;
- each route duration `<= T_max` within justified numerical tolerance;
- station may appear multiple times and on many routes.

Where possible, cross-check against the DIMACS challenge solution checker: https://github.com/ncabrera10/SolutionChecker_E-VRP . A solution is **feasible only if fully verified**; a masked action alone is never proof of global feasibility.

### 4.3 Basic heuristic baseline

Implement one compact baseline: randomized greedy **customer assignment and ordering**, with `frvcpy` (or equivalent) re-optimizing charging for every finalized route. Add simple route insertion/relocate/2-opt only when the base method works. Use a deterministic seed and a few random restarts.

**Gate:** Run the baseline on several 10- and 20-customer instances. Record success rate, summed route duration, number of routes, and runtime. If the baseline fails, fix it before creating the Transformer.

## 5. Main architecture: neural customer routing + optimized nonlinear charging

**This is the PRIMARY model/paper method.** We treat charging optimization as a trusted inner solver and learn the interdependent allocation/order of customers. This avoids asking a neural policy to rediscover known piecewise-linear charging optimization mathematics.

### 5.1 Static graph/node representation

Encode customers, depot, and charging stations using shared weights. Inputs include node type, normalized coordinates, customer service time, station technology and **charging-curve features**, e.g. breakpoint slopes and cumulative times at comparable energy fractions. Include pairwise distance/travel time/energy as relative candidate features.

Use a small node encoder (MLP + optional 1–2 attention layers), `d_model=128`, 4 heads as a reasonable **initial** setting, not a claim of optimality. A global fixed-head classifier tied to customer IDs is **not acceptable**; use per-candidate attention/pointer logits, so different customer counts work.

### 5.2 EV/route-specific Transformer — original research hypothesis

For each currently constructed route `r`, keep a visit-history sequence `H_r` and encode it using **one shared causal Transformer** (not one separately trained network per EV):

`h_r = CausalTransformer(H_r)[last]`.

A token can contain node embedding/type, arrival/departure SOC, elapsed route duration, service time, charging amount and duration if present, and a marker for depot/start. An unused new EV has a `START_EV` token.

**Subtle but crucial:** We are constructing routes **offline**. If the fixed-route optimizer is allowed to re-optimize a *planned prefix* after a new customer is appended, earlier *planned* charging stops/SOC can change. Therefore either:

- encode customer-assignment history separately from the **current optimized prefix plan**, then refresh the plan representation when it changes (preferred); or
- explicitly rebuild the affected route's tokens/Transformer state when an optimized prefix is revised.

Do **not** mislabel a mutable planned prefix as immutable physical visit history. No look-ahead information from future, unselected customers may enter the historical token. Cache Transformer output only for unchanged route histories.

### 5.3 Current state and DeepSets fleet representation

Do not require the final Transformer vector to reconstruct exact physical feasibility. Fuse it with explicit **current planned route state**:

`v_r = MLP([h_r, last_customer, optimized_prefix_duration, terminal_SOC, remaining_duration, open/closed])`.

If the prefix has multiple possible feasible charging schedules, encode the current selected (or optimized) schedule clearly and consistently. Maintain the current set `U` of unassigned customers separately.

Aggregate all route representations **permutation-invariantly**:

`g = rho(sum_r psi(v_r))` (optionally mean/max alongside sum; test before retaining).

Also encode the **remaining customers** by pooled candidate embeddings `u`, and build decision context `G = MLP([g, u, number_of_routes, number_unassigned])`.

All homogeneous routes should use shared parameters. No dependence on arbitrary EV numbering.

### 5.4 Decision process (unlimited fleet)

Use a **dynamic set of existing routes** and a `NEW_ROUTE` possibility, rather than fixing the number of vehicles. Allow a route to be marked `CLOSED` and never extended afterward. For each decision:

1. Choose an open route, or `NEW_ROUTE` (start fully charged at depot). The route selector can first use a simple deterministic strategy, then a learned pointer with a new-route token.
2. For the selected route, choose **unassigned customer `j`** using a masked pointer decoder, or `CLOSE_ROUTE` when it can return to depot.
3. Update the planned customer sequence, re-optimize charging for that prefix (or use cached plan), and update customer commitments, route state and DeepSets memory.
4. Terminate only when every customer is assigned and every route is fully feasible and closed.

For each candidate extension, use cheap necessary checks (distance/energy lower bounds, route-time lower bounds), then selectively call the fixed-route charging expert on high-scoring candidates (e.g. top-K) to verify that **prefix + candidate + depot** has an energy-feasible schedule within `T_max`. Do not mask a customer merely because the EV cannot reach the depot **directly**; several charging stops may be required.

A closed route need not have an extra depot customer token, but its finalized route must explicitly end at depot. Prevent empty routes, infinite create/close loops, duplicate customer assignments, and silent omissions. Any incomplete episode is a failure, not a feasible solution.

**Do not invent a clock-based earliest-EV scheduler.** Since this is the original static Montoya formulation with unlimited chargers, route decisions are an *offline planning order*.

### 5.5 Final charging optimization

At the end of an episode, extract each route's **ordered customer sequence**, run fixed-route nonlinear charging optimization, then independently validate the concrete solution. The main performance metric is the sum of these optimized route durations.

This gives a well-defined hybrid policy: neural assignment/order decisions + optimized charging insertions/amounts.

## 6. Direct learned-CHARGE policy — SECONDARY controlled experiment

The user's original idea also includes a special `CHARGE` token. Keep it as a **real experimental ablation**, but do **not** entangle it with the primary hybrid policy during the first working implementation.

Implement it **only after** the baseline and main hybrid model produce valid measurable results:

- First action head: unassigned customer, `CHARGE`, or `DEPOT/CLOSE`.
- If `CHARGE`: second masked pointer chooses a **reachable charging station**.
- Third head selects a **target SOC / energy amount**. Start with a small set of physically meaningful candidate charge targets (curve breakpoints and energy requirements for downstream reachable nodes). Optionally replace with a bounded continuous distribution later.
- Simulator calculates actual charging duration from the station's piecewise-linear curve. Never approximate the curve with a learned linear charging rate.
- A station is **not permanently masked because it was previously visited**; repeated station visits are allowed.
- `CHARGE` is permitted only when at least one station and positive feasible charging action exist. **Do not artificially boost its logit** just because charging is possible.
- Prevent cycling among stations and ensure routes can end; maintain a finite decision budget with clear invalid-episode handling, not a hidden change to benchmark rules.

Report **separate results** for:

**(A) Direct policy:** model selects customers, stations and charge amounts; evaluate its original realized feasible route without retrospectively changing its actions.

**(B) Hybrid policy:** model selects customer orders; charging expert chooses stations and charge amounts.

**(C) Optional direct+repair:** take the direct-policy output, fix its customer orders, and re-optimize charging; label this as a *post-optimized* variant. Do not attribute gains from charging replacement to the learned charging head.

For the primary hybrid policy, do **not** train station-head probabilities using a reward determined entirely by station choices that the optimizer later discards. Train station/quantity prediction separately from expert examples, or use the direct simulator's own reward.

## 7. Training: small, scientifically sound, actually run it

### 7.1 Data protocol

The 120 original instances are **evaluation benchmarks, not an adequate training corpus for a Transformer from scratch**.

- Generate **independent synthetic training instances**, starting at 10 and 20 customers, then 40 and 80; larger sizes only after smaller ones work.
- Match broad observed properties of Montoya (random/clustered/mixed customer geometry, station densities/placement, service times, route limits, battery and station technologies), using **new random seeds and geometry**, not coordinate permutations of benchmark instances.
- Use a **route-first or verified constructive generator** where practical; check feasibility with the charging expert and simulator. Do not treat arbitrary generated instances as solvable without checking.
- Keep a separate synthetic validation seed set; use official instances only for smoke-checks and held-out evaluation, clearly indicating any benchmark instances inspected during development.
- Maintain no hidden overlap or leakage between generated train/validation and published benchmark coordinates.

### 7.2 Initial training

Start small: `d_model=128`, 2 causal Transformer layers, 4 heads, DeepSets MLP, masked pointer decoder. Use a shared policy across all routes. Use batch sizes that fit hardware and train on 10-customer instances first.

**Warm start:** Generate teacher decisions from the heuristic + fixed-route charging optimizer. Use behavior cloning if it materially improves learning. Many different route-construction orders describe the same final solution; avoid encoding arbitrary route IDs as ground truth. Randomize valid equivalent construction orders when reasonable.

**Fine-tune:** Use a **simple self-critical policy gradient / REINFORCE with greedy or moving-average baseline** first. Full PPO is optional only after this trains and shows an evidence-based need. The global objective is total verified route duration; valid complete solutions should be optimized; invalid/incomplete solutions receive an explicit penalty and are counted in feasibility rate. Respect masks while calculating policy log-probabilities.

Recommended reward idea:

`reward = - total_duration` for complete feasible solutions; otherwise a clearly specified penalty that discourages omissions/dead ends. Do not compare raw objective values across different customer counts without normalization/stratification. Never call a partial route feasible just because its current SOC is nonnegative.

**Debug expected learning:** Save episode reward, feasible episode rate, greedy validation duration, route count, policy entropy, training time and decision count. Plot at least training vs validation objective and feasibility; if learning collapses, investigate infeasibility/masks/reward scales before adding model layers.

### 7.3 Training and evaluation budget

Start with 1–2 small instances to prove correctness, then a small baseline set, then a short smoke training (e.g. 100–500 updates) with real logs. Only lengthen training if curves justify it. Fix random seeds for comparisons; ideally report 3 training seeds for the final ablation if compute permits.

When evaluating, hold **inference time budget and number of sampled solutions** constant between models. Measure total wall time including expensive calls to the charging expert and local search, not merely neural forward passes.

## 8. Research experiments and ablations

Run comparisons on **identical instances** and objective rules, with the same search budget:

| ID | Routing policy | Fleet information | Charging |
|---|---|---|---|
| H0 | Randomized greedy + optional local search | Rule-based | Fixed-route oracle |
| H1 | Neural pointer, **current route state only** | Unassigned set only | Fixed-route oracle |
| H2 | History Transformer | No DeepSets | Fixed-route oracle |
| H3 | State-only neural pointer | DeepSets over current route states | Fixed-route oracle |
| H4 | **History Transformer + DeepSets** (main hypothesis) | History-aware fleet memory | Fixed-route oracle |
| H5 | H4 + one vehicle/route cross-attention | History-aware + pairwise | Fixed-route oracle |
| H6 | H4 with direct `CHARGE`/station/quantity heads | Same fleet memory | Learned, exact simulator |
| H7 | H6 customer sequences post-optimized | Same fleet memory | Charging oracle after generation |

H0–H4 are the **must-have** experiments. H5–H7 are conditional on results, time, and correctness. Do not claim historical memory is beneficial without H1–H4. The original problem is deterministic and close to Markov given a complete current planning state; **history may not help**, and that negative result is scientifically meaningful.

Evaluate by benchmark customer-size groups (10, 20, 40, 80, 160, 320) as support becomes available. For each group report:

- number of attempted instances;
- **full-solution feasibility rate**;
- mean/median sum-of-route-duration over feasible solutions (not over failed runs);
- gap to published **verified** BKS when obtainable, defining `(our_cost - BKS)/BKS * 100%`;
- number of used routes, charging stops, total charging time;
- wall-clock inference time, charging-expert calls, and any local-search time;
- aggregate results by distribution/instance class if possible.

Use `NA` for unavailable BKS; do not invent a baseline. Report failures separately to avoid survivorship bias. For final runs, create a CSV with **one row per instance/seed/method**, a compact summary table, and a few interpretable learning-curve / route plots. No elaborate dashboard.

## 9. Feasibility and correctness sanity checks (small, not a testing framework)

At a minimum demonstrate:

1. All parsed station curves interpolate monotonically; `C_s(Q)` matches the final breakpoint; charging from `q` to same `q` takes zero time.
2. A fixed route's energy/time recomputed independently matches the oracle and/or published example within explained tolerance; service time counted exactly once.
3. Full-battery depot departure, no overcharging, no negative SOC, no customer twice, no unserved customers in successful solutions.
4. Feasible return paths may have **multiple** charging stops; station revisits and visits by other EVs are allowed.
5. Routes exceeding `T_max` are rejected even if they are energy-feasible.
6. New-route action is possible without a user-imposed fleet cap; no empty routes.
7. Invalid/incomplete solutions are never included in means of feasible objective values.

These may be a short `sanity` command in `evaluate.py` or direct assertions; **do not set up hundreds of isolated tests**.

## 10. Concrete execution order and stop/go gates

Work *sequentially*; do not build all modules before your first execution.

**Milestone M0 — Dataset understood:** inspect and summarize real XML; confirm exact cost/unit semantics. Output dataset summary and explain any unusual fields.

**M1 — Correct core:** parser, curve evaluator, independent route validator, fixed-route oracle integration. Demonstrate a few verified feasible routes. **No neural code yet.**

**M2 — Real baseline:** randomized greedy construction with optimized nonlinear charging; run on representative 10/20-customer instances. Print actual feasibility/objective/runtime. Improve this until it gives useful reference solutions.

**M3 — First learned model:** implement neural pointer with state-only input (H1), synthetic generator, short training/evaluation. Demonstrate reward or validation trends against H0.

**M4 — Research hypothesis:** add shared visit-history Transformer + DeepSets (H4), trained and evaluated on identical instances. Run H2/H3 ablations, plus 40-customer generalization if reasonable.

**M5 — Original charging idea:** only after M4 is working, add direct `CHARGE`/station/quantity model and compare with oracle-assisted version. Keep the benchmark formulation unchanged.

**M6 — Strengthen results:** extend to 80/160/320 instances within hardware/time budget, tune only based on validation, optionally add light local search or route attention, prepare a compact research results table.

At **each** completed milestone, write in `results/notes.md`:

- what works;
- exact command run;
- actual numeric results and hardware/runtime;
- key failure or limitation;
- next **single highest-value** experiment.

If the environment does not permit training, explicitly identify the blocker and leave an executable minimal command and no fake plots/results. If `frvcpy` fails on current Python, try a compatible environment or a transparently labeled alternative; do not silently skip charging feasibility.

## 11. Commands and artifacts expected from you

Make it possible to run a compact set of commands similar to the following (adapt to your actual argparse implementation and document the real syntax):

```bash
python evaluate.py --mode inspect --data dataset
python evaluate.py --mode sanity --data dataset
python evaluate.py --mode baseline --size 10 --limit 5 --seed 42
python train.py --model state --size 10 --steps 500 --seed 42
python train.py --model history_deepsets --size 10 --steps 500 --seed 42
python evaluate.py --mode compare --size 10 --seed 42
```

Do not add pretend CLI flags or claim these example commands were executed if your final implementation uses different commands. Save best checkpoints by feasible validation quality, report benchmark evaluations with the model/checkpoint and seed used, and record `requirements.txt` versions.

**First progress response to me (after inspecting files):** tell me what was found in `dataset/`, which core assumptions were confirmed, and which small instance will be used for the first executable baseline. **Then continue coding and running**—do not wait for approval unless blocked.

**At the end of each research session:** succinctly report **what was implemented, what was actually run, numerical results, problems, and the next experiment**. Keep explanations concise; prioritize the empirical work.

## 12. References — use for correctness, not for building a literature-review app

- Montoya et al. (2017), *The electric vehicle routing problem with nonlinear charging function*, Transportation Research Part B, DOI: https://doi.org/10.1016/j.trb.2017.02.004
- Montoya benchmark, 120 VRP-REP instances: https://www.vrp-rep.org/datasets/item/2016-0020.html
- DIMACS official E-VRP-NL / E-VRP-NL-C formulation: https://dimacs.rutgers.edu/programs/implementation-challenges/vehicle-routing/implementation-challenge-vehicle-routing/electric-vehicle-routing-problem
- Froger et al. (2019), *Improved formulations and algorithmic components for the electric vehicle routing problem with nonlinear charging functions*: https://www.sciencedirect.com/science/article/pii/S0305054818303253
- `frvcpy` fixed-route optimal charging solver: https://github.com/e-VRO/frvcpy
- DIMACS E-VRP solution checker: https://github.com/ncabrera10/SolutionChecker_E-VRP
- Deep Sets (Zaheer et al. 2017): https://arxiv.org/abs/1703.06114
- Attention, Learn to Solve Routing Problems! (Kool et al. 2019): https://arxiv.org/abs/1803.08475

**Final priority:** A small scientifically correct and **executed** research project is worth more than a large polished repository with no valid trained models, no trustworthy feasible solutions, and no comparative numbers.
