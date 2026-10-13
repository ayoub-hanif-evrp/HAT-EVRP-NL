# Nonlinear Charging Study (RQ3)

Priority experiment: consequences of replacing Montoya **nonlinear** charging
with explicitly defined **linear** surrogates, with every plan re-evaluated under
**true** piecewise-linear physics.

Coverage: **20/20 official instances** for each of \(n\in\{10,20,40\}\) (full size groups).

Raw data: `results/raw/nonlinear_charging/`  
Main table: `results/tables/table15_nonlinear_vs_linear_charging.md`  
Figures: `fig_main02_*`, `fig_main03_*`, `fig_main04_*`

---

## 1. True charging model

For station technology \(s\), let \(F_s(q)\) be the cumulative charging time needed to
reach battery energy \(q\) on the Montoya piecewise-linear curve. A charge from
SOC \(q_\mathrm{in}\) to \(q_\mathrm{out}\) takes

\[
T_\mathrm{charge} = F_s(q_\mathrm{out}) - F_s(q_\mathrm{in}).
\]

Full physically validated route duration:

\[
Z_\mathrm{full} = T_\mathrm{travel} + T_\mathrm{service} + T_\mathrm{nonlinear\_charge}.
\]

Literature-compatible objective (service subtracted for BKS comparison):

\[
Z_\mathrm{lit} = T_\mathrm{travel} + T_\mathrm{nonlinear\_charge}.
\]

Feasibility always uses service time and the true maximum route duration.

---

## 2. Linear surrogates (not interchangeable)

| Model | Definition | Interpretation |
|---|---|---|
| **TRUE-NL** | Station-dependent Montoya PWL curves; `frvcpy` FixedRouteOracle | Physically validated baseline |
| **LINEAR-SECANT** | Constant rate on the empty→full chord for each technology | Average rate over full charge; **pessimistic** vs early SOC on concave curves |
| **LINEAR-EARLY-RATE** | Constant rate from the first true breakpoint segment, extrapolated to \(Q\) | **Optimistic** for concave curves; underestimates late-SOC charging time |

Surrogates replace `Instance.curves` only for **planning**. Original XML is never modified.
Re-evaluation always uses the original Montoya curves via the Python physical checker.

---

## 3. Experiment A — Fixed-route charging comparison

**Protocol**

1. Build customer sequences with History Transformer Routing (H2) Cycle-3 + basic LS under TRUE-NL.
2. Optimize charging independently under each model for those **fixed** sequences.
3. Re-evaluate every planned stop list with the true nonlinear checker.
4. Separately report TRUE-NL re-optimization of the same sequences after a linear plan
   (`repair_TRUE-NL_after_*`) — **not** credited to the original linear plan.

### Feasibility under true physics (plan methods only)

| n | TRUE-NL | LINEAR-SECANT | LINEAR-EARLY-RATE |
|---:|---:|---:|---:|
| 10 | **20/20** | 16/20 | 16/20 |
| 20 | **20/20** | 12/20 | 12/20 |
| 40 | **20/20** | 11/20 | 11/20 |

**Key finding.** Linear planning produces plans that frequently become **infeasible under
true Montoya physics**, especially as \(n\) grows. At \(n=40\), only **11/20** linear-planned
instances remain feasible under true physics, while TRUE-NL remains **20/20**.

- **LINEAR-EARLY-RATE** often appears feasible under the optimistic surrogate
  (e.g. 20/20 planner-feasible at \(n=40\)) but fails true-physics duration checks
  (9 duration violations at \(n=40\)).
- **LINEAR-SECANT** is more conservative: many instances already fail under the planner
  (`missing_stops` / no feasible FRVCP plan), so fewer optimistic false-feasible plans
  survive to true-physics evaluation — yet true-physics feasibility is still only 11/20 at \(n=40\).
- **Repair with TRUE-NL** restores 20/20 feasibility in all size groups; this is a
  *separate* method and must not be used to claim the linear plan was feasible.

Do **not** compare feasible-only mean durations across methods with unequal feasibility rates.

---

## 4. Experiment B — End-to-end routing sensitivity

Same pretrained H2 checkpoint (`H2_n10_seed42_c3.pt`) and construction seed 42;
only the charging curves passed to `frvcpy` during construction change.
All solutions are scored under TRUE-NL.

| n | TRUE-NL true-feas | LINEAR-SECANT | LINEAR-EARLY-RATE |
|---:|---:|---:|---:|
| 10 | 20/20 | 19/20 | 15/20 |
| 20 | 20/20 | 19/20 | 17/20 |
| 40 | 20/20 | 11/20 | 9/20 |

Misspecifying the charging law changes route sequences, charger selection, and
feasibility. Decision-change fingerprints are in
`end_to_end_decision_changes.csv`.

---

## 5. Why nonlinear charging matters in practice

1. **Feasibility:** Linear assumptions yield plans that violate true duration/SOC constraints
   when re-evaluated on Montoya curves (up to **45%** of \(n=40\) fixed-route linear plans;
   up to **55%** of \(n=40\) end-to-end EARLY-RATE plans).
2. **Optimistic bias:** EARLY-RATE underestimates late-SOC charging time → planner accepts
   charge quantities that are too slow in reality.
3. **Pessimistic / planner failure:** SECANT can make FRVCP declare sequences infeasible
   even when TRUE-NL would find a feasible charge schedule.
4. **Repair is not free credit:** Re-optimizing with TRUE-NL after a linear plan restores
   feasibility but is a different algorithm.

This directly supports Montoya et al. (2017): neglecting nonlinearity can produce
infeasible or expensive routes.

---

## 6. Commands

```bash
python scripts/experiments/run_nonlinear_charging_study.py --stage all
python scripts/reporting/charging_and_reward_figures.py
```

---

## 7. Limitations

- Sequences for the fixed-route study come from one H2+LS seed (42), not an exhaustive
  set of all feasible sequences.
- End-to-end uses frozen neural weights; we isolate charging-model effects, not joint
  re-training under linear physics.
- We do not claim that every linear approximation in the literature equals SECANT or EARLY-RATE.
