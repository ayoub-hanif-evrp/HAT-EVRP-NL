# Training and Reward Methodology

Grounded in `train.py`, `model.py`, Cycle-3 training logs, and checkpoint payloads.
Optional training flags are described only where actually used.

## 1. Problem and hybrid evaluation

The hierarchical policy proposes customer sequences (and route assignments).
For every candidate sequence, **`frvcpy` solves the fixed-route nonlinear charging
subproblem** (FRVCP-NL). An independent Python forward simulator
(`check_route` / `check_solution`) validates SOC, duration, and coverage.

`frvcpy` is **algorithmic and non-differentiable**. It is not trained.

Full duration:

\[
Z_\text{full} = \text{travel} + \text{service} + \text{nonlinear charging}.
\]

Nonlinear charging for a visit is \(F_s(q_\text{out}) - F_s(q_\text{in})\).
Service is constant once all customers are served; literature-compatible
duration is \(Z_\text{lit} = Z_\text{full} - \sum \text{service}\).

## 2. Architectures (internal keys unchanged)

| Key | Display name | History | DeepSets |
|---|---|---|---|
| H1 | State Pointer | no | no |
| H2 | Route Transformer (RT) | yes | no |
| H3 | Fleet DeepSets | no | yes |
| H4 | Route Transformer + DeepSets (HAT-EVRP-NL) | yes | yes |

RT encodes optimizer-enriched route-stop sequences (node embedding plus SOC, energy, and charging features). A token is a stop representation, not necessarily one committed historical action.

Defaults: `d_model=128`, `layers=2`, `heads=4` (`PolicyConfig` in `model.py`).

The primary training entry point is `python train.py --pipeline primary`. It runs only behavior cloning, then REINFORCE (`bc_rl.train_primary`). POMO and repair-RL remain on `--pipeline legacy` for historical reproduction and are refused by the primary path.

Normalized reward (primary default):

\[
R_{\text{feasible}} = -Z_{\text{full}} / (n T_{\max}), \quad
R_{\text{infeasible}} = -1 - \epsilon - \alpha \, u - \beta \, v
\]

where \(u\) is the unserved fraction when known and \(v\) is a normalized violation only if the decode result already carries a finite `normalized_violation`. The legacy reward \(R=-Z_{\text{full}}\) or \(-100\) remains available with `--reward legacy`. Checkpoint selection uses feasibility first, then the mean duration in which infeasible evaluations count as \(n T_{\max}\) rather than being dropped.

## 3. Training pipeline (what Cycle-3 actually used)

### A. Independent synthetic instance generation

`make_teacher_data` builds fresh synthetic E-VRP-NL instances from a seeded
generator (`generator` default `v2` for Cycle-3). Train/validation seed bands
are disjoint (`700000+` / `800000+` offsets).

**Cycle-3:** `--size 10 --train-instances 120 --val-instances 12`.

### B. Teacher construction with frvcpy

Expert routes come from greedy / local-search teachers; each route is charged
with `frvcpy`. Teachers supply hierarchical route/customer decisions for BC.

### C. Behavior cloning (used for all Cycle-3 H1–H4)

Masked hierarchical cross-entropy on teacher route and customer actions
(`teacher_loss`). **400 BC steps**, `eval_every=40`.

### D. Optional multi-target expert action learning

Flag `--multi-target`. **Not used** for Cycle-3 primary checkpoints.

### E. Optional fleet-counterfactual DeepSets training

Flag `--fleet-counterfactual`. **Not used** for Cycle-3 primary checkpoints
(used in some post-Cycle-3 `perf_v2b_fc` experiments only).

### F. Standard REINFORCE (used for Cycle-3)

After BC, **40 RL steps** with `pomo_k=0`:

- Sampled rollout vs greedy rollout baseline.
- \(R = -Z_\text{full}\) if feasible else \(-100\).
- \(L = -(R - R_\text{greedy}) \sum \log \pi(a_t)\).

### G. POMO-style multi-start REINFORCE

`pomo_reinforce_step` with shared mean baseline over K forced first customers.
**Not used** for Cycle-3 (`pomo_k=0`). Present in some `perf_*_pomo` runs.

### H. Position-aware neural repair REINFORCE

`repair_reinforce_step`: \(R = Z_\text{before}-Z_\text{after}\) or \(-50\) if infeasible;
baseline = mean repair-rollout reward. **Not used** for Cycle-3
(`repair_rl_steps=0`). Used in later insertion/repair experiments.

### I. Validation and checkpoint selection

At logged evaluation steps, greedy decode on the **held-out synthetic** set
scores `(feasible_rate, -validation_objective)`. The **best** state is restored
before saving the `.pt` file — **not** necessarily the final RL iterate.

## 4. Reward subsection (exact implementation)

### Standard REINFORCE / POMO

| Outcome | Reward |
|---|---|
| Feasible | \(R = -Z_\text{full}\) |
| Infeasible | \(R = -100\) |

Baseline:

- Standard: greedy rollout reward.
- POMO: mean reward of K rollouts.

Advantage \(A = R - b\). Policy-gradient loss
\(L_\text{RL} = -A \sum_t \log \pi(a_t)\) (mean over POMO rollouts when used).

### Repair RL

| Outcome | Reward |
|---|---|
| Feasible repair | \(R = Z_\text{before} - Z_\text{after}\) |
| Infeasible | \(R = -50\) |

Baseline: mean of repair-rollout rewards.

### Distinctions

| Quantity | Role |
|---|---|
| Reward \(R\) | Scalar feedback for the sampled trajectory |
| Baseline \(b\) | Control variate (greedy or mean) |
| Advantage \(A\) | \(R-b\); zero ⇒ no update |
| \(L_\text{RL}\) | Policy-gradient loss |
| BC loss | Cross-entropy (not a reward) |
| Validation objective | Mean feasible \(Z_\text{full}\) on synth val |
| Feasibility rate | Fraction of feasible val decodes |

Over feasible complete solutions, maximizing \(R=-Z_\text{full}\) is equivalent to
minimizing \(Z_\text{lit}\) because service is constant.

**Penalty caveat.** Fixed \(-100\) / \(-50\) were not re-tuned by problem size.
All-infeasible repair batches can produce near-zero advantages and little learning.
Rewards were **not** changed in this presentation task.

Historical Cycle-3 CSVs **do not** store sampled reward distributions; do not
invent them from validation objectives. Controlled RQ4 runs write per-step
`reward_*.csv` with `mean_sampled_reward`, `baseline_reward`, `mean_advantage`,
rollout feasibility, grad norm, etc. (`results/raw/reward_learning/`).

## 5. What Cycle-3 logs show (seed 42 example)

| Model | BC end val h | BC best val h | RL end val h |
|---|---:|---:|---:|
| H2 | 15.7947 | ~15.71 | **15.4834** (improved) |
| H4 | 15.7137 | ~15.71 | **31.0154** (collapsed) |

Across seeds 42–44, RL endpoints beat best-BC validation on **1/3 H2 seeds** and
**0/3 H4/H1/H3 seeds**. H4’s RL stage is especially unstable (seed 42 collapses
to ≈31 h). Checkpoint selection keeps the best validation score, so **deployed
H4 Cycle-3 weights are BC-selected**. H2 seed 42’s RL endpoint is better than
BC-best, but that is not a robust multi-seed finding.

**Conclusion:** Cycle-3 synthetic-val endpoints do **not** support claiming that RL
improves deployed HAT-EVRP-NL (H4), nor a general claim that RL reliably improves H2.

### Controlled RQ4 (separate; does not overwrite Cycle-3)

Fresh `*_bc_only_rq4.pt` → 40-step REINFORCE → `*_rl_from_bc_rq4.pt`, evaluated
construction-only on official n=10:

- H2: RL better on **1/3** seeds (mean BC 22.157 h vs RL 22.880 h).
- H4: RL better on **3/3** seeds (mean BC 22.477 h vs RL 21.636 h).

These numbers are **not** Cycle-3 fair-LS quality. See
`results/REWARD_AND_LEARNING_STUDY.md`.

## 6. Training size vs evaluation size

All primary training is on **synthetic n=10**. Official Montoya n=20/40/80
results are **transfer evaluations**, not training at those sizes.

## 7. Post-Cycle-3 experiments (separate)

Warm-start `perf_*` checkpoints may add POMO, fleet-counterfactual, or repair RL.
They are **not** the Cycle-3 primary story and must be labeled separately.

## Commands

```bash
# Cycle-3 style (reference)
python train.py --model H2 --size 10 --steps 400 --rl-steps 40 \
  --train-instances 120 --val-instances 12 --eval-every 40 --seed 42 --version c3

# Controlled BC vs RL (does not overwrite *_c3.pt)
python scripts/experiments/run_bc_rl_controlled.py --stage all --models H2,H4 --seeds 42,43,44

# Derive learning / charging tables/figures
python scripts/reporting/analyze_training_logs.py
python scripts/reporting/charging_and_reward_figures.py
python scripts/reporting/build_results.py --all --skip-decomp
```
