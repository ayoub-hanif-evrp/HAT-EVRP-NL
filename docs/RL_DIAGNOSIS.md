# Why REINFORCE has not improved behavior cloning

Primary question: does REINFORCE improve the routing decisions learned by behavior cloning when the solution is scored by exact nonlinear charging, with no local search and no ruin-and-repair?

The original 20 runs and the later shuffled-sampler runs are unchanged. This note reads their logs. It does not treat hybrid insertion or classical search as an RL result.

## What the logs show

Each RL phase draws 2,400 rollouts (300 updates, batch size 2, 4 rollouts). Across the 20 original runs that is 48,000 rollouts. Across the 10 shuffled runs it is another 24,000. Every recorded rollout is feasible. The infeasibility penalty is never the training signal. The reward that is used is `R = -Z_full / (n T_max)`.

Entropy falls in every run. On the original Route Transformer seeds it moves from about 0.60–0.71 at the first update to about 0.44–0.52 at update 300. The entropy coefficient in those runs is zero, so the drop comes from the policy gradient itself.

Validation duration is worse at update 300 than at the behavior-cloning checkpoint in all 20 original runs. The end-of-training increase is about +0.04 h to +0.51 h. For Route Transformer, the best validation point is the first check, at step 25, on all five seeds, and only one of the twelve checks beats the behavior-cloning start. Those wins are a few hundredths of an hour. Checkpoint selection is doing what it should: when the post-RL validation is not better, the deployed checkpoint stays at behavior cloning. That is not an RL improvement.

The logged training reward cannot be read as learning. The cyclic sampler changes which instances appear, so a higher mean reward at step 300 is a different pair of instances, not the same objective getting better.

The gradient norm stays around 0.05–0.12 and the mean absolute loss around 0.04. The updates are large enough to concentrate the policy and small enough that the greedy validation policy does not become a better constructor.

## Failure mechanism

The consistent mechanism is drift away from the behavior-cloning initialization. The sampled advantages are differences among already feasible routes. They are not large enough, under the normalized reward and four leave-one-out rollouts, to produce a stable reduction in greedy validation duration. What does accumulate over 300 updates is a less entropic policy whose greedy construction is worse.

Covering the 199 training instances that the cyclic index skipped does not remove this pattern. The shuffled runs also end with lower entropy and, on 9 of 10 seeds, a worse validation duration than behavior cloning.

## What the new logs add about the signal

The KL runs record the within-update reward spread that the older CSVs did not. Across the ten runs the mean within-batch reward standard deviation is 0.0185. Because the normalized reward divides duration by `n T_max = 100`, that is about 1.85 h of spread among the four rollouts of an instance. The mean absolute leave-one-out advantage is 0.0216, about 2.2 h. Sampled routes are not copies of one another. The missing piece is not a reward that never varies.

Every rollout in these runs is also feasible. Entropy still falls, for example from 0.70 to 0.57 on Route Transformer seed 42 and from 0.97 to 0.39 on seed 46.

## The first penalty did not have the KL gradient

`results/raw/bc_rl_kl_v1` added `log π_θ(a) - log π_BC(a)` for the sampled action. Sampling is discrete, so PyTorch does not differentiate through the action. The expected gradient of that term, given the state, is zero. The recorded validation failure therefore does not show that a real KL penalty fails. Those files stay as the record of that incorrect term.

On that incorrect term, validation duration minus behavior cloning was +0.068 h for Route Transformer (lower on 1/5 seeds) and +0.125 h for RT + DeepSets (lower on 0/5 seeds). Those numbers stay in `results/raw/bc_rl_kl_v1`. They are not evidence about a KL regularizer.

## The corrected intervention

REINFORCE stays the update. The added term is now `0.01 * KL(π_θ || π_BC)` on the masked route distribution and the masked customer distribution at each decision. The behavior-cloning weights are frozen and receive no gradient. Entropy regularization stays off. The learning rate, reward, rollout count, and 300-update budget stay as before. Sampling is the shuffled order. Runs start from the same saved `bc_best.pt` files for Route Transformer and RT + DeepSets, seeds 42–46, and write to `results/raw/bc_rl_kl_v2`.

Official neural-construction evaluation starts only if validation improves on at least four of five seeds and the paired interval excludes zero. That rule was not met. Route Transformer validation duration minus behavior cloning is +0.045 h, interval [−0.109, +0.199], lower on 2/5 seeds. RT + DeepSets is +0.049 h, interval [−0.076, +0.174], lower on 2/5 seeds. Means are 21.475 h to 21.520 h, and 21.715 h to 21.764 h. The files are `results/raw/bc_rl_kl_v2`. Official evaluation was not started. This is a negative result for this coefficient and this budget. It does not justify a larger coefficient chosen after seeing these numbers.

## What this does not claim

It does not claim that reinforcement learning improves the neural constructor. The original paired construction result remains −0.091 h for Route Transformer, interval [−0.336, +0.154], which includes zero. RT + DeepSets is worse on average after the original RL phase. Hybrid insertion and classical ruin-and-repair are separate from this question.
