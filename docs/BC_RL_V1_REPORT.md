# BC → RL v1 — what was implemented and what was actually run

Branch: `bc-rl-v1` (from `f4494d0de99395ffbabb272e45615957ce38c347`).
Published at https://github.com/ayoub-hanif-evrp/HAT-EVRP-NL/tree/bc-rl-v1. That push did not update `main`. `main` was later fast-forwarded to `7671f6411a37666212656121d02a7c3af41ce1f7`, which is also the tip of `bc-rl-v1`.

## Pipeline

Primary training is only behavior cloning, then REINFORCE:

`python train.py --pipeline primary ...`

`--pipeline legacy` keeps the historical loop, including optional POMO and repair-RL. The primary path refuses both.

## Losses and rewards

BC loss is unchanged: hierarchical cross-entropy (`teacher_loss`). `--multi-target` and `--fleet-counterfactual` stay inside BC. They were not turned on in the smoke run.

Normalized RL reward (default):

- Feasible: \(R = -Z_\mathrm{full}/(n T_\max)\). With at most \(n\) routes each of duration at most \(T_\max\), this lies in \([-1, 0]\).
- Infeasible: \(R = -1 - \epsilon - \alpha u - \beta v\), with \(\epsilon=0.05\). \(u\) is the unserved fraction when `n_unserved` is known. \(v\) is included only if the result already has a finite `normalized_violation`. Missing violations are not invented.

Legacy \(R=-Z_\mathrm{full}\) or \(-100\) remains `--reward legacy`.

REINFORCE uses detached advantages. Leave-one-out baselines exclude the rollout itself. One rollout falls back to advantage 0. Entropy is a differentiable tensor on the decode result; it enters the loss only when `--entropy-coef` is positive. The smoke run used coefficient 0, so entropy regularization was not active.

`frvcpy` stays outside the gradient. Checkpoint selection ranks feasibility first, then mean duration. Infeasible validation cases count as \(n T_\max\) instead of being dropped from the mean. Artifacts per run: `bc_best.pt`, `rl_best.pt`, `deployed_best.pt`. Deployment stays on BC when RL does not improve the validation score.

## Names

Reader-facing names now come from `presentation_names.py`: State Pointer, Route Transformer (RT), Fleet DeepSets, Route Transformer + DeepSets (HAT-EVRP-NL). Internal keys H1–H4 and Cycle-3 checkpoint files are unchanged. RT tokens are optimizer-enriched stop features, including charging information. They are not claimed to be one token per committed action.

## Executed experiment

Smoke only. Not a scientific comparison.

| Item | Value |
|---|---|
| Run | `results/raw/bc_rl_v1/H2_n6_seed7_bcrl_smoke_20261018T225523Z` |
| Model | H2 (Route Transformer), seed 7, CPU |
| Data | 2 synthetic train instances, 1 validation instance, generator v2, n=6 |
| BC | 2 steps, lr 1e-3 |
| RL | 1 step, lr 1e-4, 2 LOO rollouts, normalized reward |
| Validation | feasible 1/1, penalized mean 14.530 h before and after the RL step |
| RL improved | no |
| Deployed | BC |
| RL oracle calls | 29 on that single update |

An earlier directory `...225447Z` is the same schedule recorded before instance-seed parsing was corrected. Do not treat either run as evidence that RL helps or hurts RT.

Unit tests: `tests/test_bc_rl.py` passed (11). Full `unittest discover`: 89 tests, 1 caption-regex failure after the rename, then fixed. Historical `H2_n10_seed42_c3.pt` still loads.

## Not executed (needs authorization)

These were not run. No official Montoya BC-versus-RL table was produced, and no new main-paper figure claims a BC/RL gain.

1. Larger-data BC comparison versus the 120/12 Cycle-3 budget.
2. Matched multi-target and fleet-counterfactual BC ablations.
3. Legacy versus normalized reward ablation.
4. `cost_lambda` / `route_lambda` validation sweep, then a frozen official test.
5. Five seeds × State Pointer, RT, Fleet DeepSets, and RT+DeepSets, with construction, fair local search, and hybrid insertion.

Cycle-3 training for H2 seed 42, after teachers were built, logged about 101 s on CPU for 400 BC steps and 40 single-sample RL steps. Multiplying rollouts, models, and seeds, plus teacher generation and official evaluation, is a multi-hour CPU job. It should be scheduled explicitly rather than inferred from the smoke test.

## Classification

| Claim | Status |
|---|---|
| Primary loop is BC then RL | implemented and smoke-tested |
| RL initialized from best BC and did not replace it when validation did not improve | shown on the n=6 smoke only |
| RL improves Route Transformer or HAT-EVRP-NL | not tested at research scale; not claimed |
| DeepSets beats RT | not retested; historical Cycle-3 conclusion stands |
| Historical Cycle-3 numbers | unchanged |
