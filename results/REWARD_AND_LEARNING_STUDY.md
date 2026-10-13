# Reward and Learning Study (RQ4)

Controlled BC-only versus standard REINFORCE fine-tuning with **real** reward logs.
Cycle-3 checkpoints (`*_c3.pt`) are **not** overwritten.

Raw: `results/raw/reward_learning/`  
Methodology: `docs/TRAINING_AND_REWARD_METHODOLOGY.md`  
Table: `results/tables/table06_bc_vs_rl.md`  
Figure: `results/figures/fig_main07_reward_dynamics.png`

---

## 1. Implemented reward (code-grounded)

### Standard REINFORCE

\[
R(\pi)=\begin{cases}-Z_\mathrm{full}(\pi) & \text{feasible},\\ -100 & \text{infeasible.}\end{cases}
\]

\[
L_\mathrm{RL}=-(R-b)\sum_t\log\pi_\theta(a_t\mid s_t),\quad b=\text{greedy rollout reward}.
\]

### Behavior cloning

\[
L_\mathrm{BC}=-\sum_t\log\pi_\theta(a_{\mathrm{expert},t}\mid s_t)
\]

with hierarchical route/customer factorization. `frvcpy` is non-differentiable;
gradients use sampled routing log-probabilities after fixed-route optimization.

**Reward ≠ loss.** BC CE and PG losses are not comparable; validation \(Z_\mathrm{full}\) is separate.

---

## 2. Controlled RQ4 experiment (executed)

**Protocol** (`scripts/experiments/run_bc_rl_controlled.py`)

1. BC-only: 400 steps, `rl_steps=0` → `checkpoints/*_bc_only_rq4.pt`
2. Warm-start RL: `steps=0`, `rl_steps=40`, standard REINFORCE → `*_rl_from_bc_rq4.pt` + `reward_*.csv`
3. Evaluate greedy **construction-only** on all 20 official n=10 instances
4. Models H2, H4; seeds 42, 43, 44; generator `v2`; 120/12 train/val

### Official n=10 construction-only (full duration h)

| Model | Seed | BC-only | RL-from-BC | RL better? |
|---|---:|---:|---:|---|
| H2 | 42 | 22.5423 | 23.4298 | no |
| H2 | 43 | 21.4002 | 21.1998 | **yes** |
| H2 | 44 | 22.5303 | 24.0106 | no |
| H4 | 42 | 22.6227 | 22.0479 | **yes** |
| H4 | 43 | 22.6678 | 21.4805 | **yes** |
| H4 | 44 | 22.1390 | 21.3808 | **yes** |

**Seed-level summary**

- H2: RL improves official construction mean on **1/3** seeds; mean over seeds BC 22.157 h vs RL 22.880 h (RL worse on average).
- H4: RL improves on **3/3** seeds under this controlled construction-only protocol; mean BC 22.477 h vs RL 21.636 h.

All rows are 20/20 feasible. These are **not** Cycle-3 fair-LS numbers and must not be mixed with them.

### Reward logs

Per-step CSVs include: step, sampled reward, greedy baseline, advantage, rollout feasibility,
sampled full h, PG loss, entropy, grad norm, learning rate. See `reward_H*_rl_from_bc_rq4.csv`.

---

## 3. Historical Cycle-3 log analysis (synthetic validation)

Not equivalent to the controlled BC-only vs RL checkpoint bake-off above.

| Model | Mean best-BC val (h) | Mean RL endpoint (h) | Seeds RL end < best BC |
|---|---:|---:|---:|
| H1 | 15.5464 | 17.8534 | (see derived) |
| H2 | 15.4235 | 22.0845 | **1/3** |
| H3 | 15.6168 | 22.0632 | (see derived) |
| H4 | 15.4325 | 26.4013 | **0/3** |

Across the three Cycle-3 training seeds, reinforcement-learning endpoints do not
demonstrate consistent improvement over the best behavior-cloning validation
checkpoints on the synthetic validation set. This effect is especially evident in
the Transformer–DeepSets configuration, for which all three RL endpoints are worse
than the corresponding best BC validation result. Deployed Cycle-3 checkpoints use
synthetic-validation selection, preventing poor RL endpoints from automatically
replacing better pretrained weights.

---

## 4. Interpretation (do not overclaim)

1. **Controlled official construction-only** (fresh BC → 40 RL steps): H4 shows consistent
   seed-level improvement; H2 does not (1/3).
2. **Historical Cycle-3 synthetic-val endpoints**: RL often *degrades* vs best BC, especially H4 (0/3).
3. These protocols differ (data split metric, checkpoint lineage, evaluation set, search).
   Neither alone justifies “RL always helps” or “RL never helps.”
4. Distinguishing last BC / best BC / final RL / best deployed remains essential.

---

## 5. Reward design risks

- If feasible \(Z_\mathrm{full}>100\), infeasible \(R=-100\) can outscore that feasible rollout.
- Large grad norms appear in reward CSVs (unstable PG steps).
- Alternative scalings require new controlled experiments; historical rewards are not rewritten.

---

## 6. Placeholder figure policy

`fig14_reward_dynamics_unavailable.png` is archived when real dynamics exist.
Publication figure: `fig_main07_reward_dynamics.png`.

---

## 7. Commands

```bash
python scripts/experiments/run_bc_rl_controlled.py --stage all --models H2,H4 --seeds 42,43,44
python scripts/reporting/charging_and_reward_figures.py
```
