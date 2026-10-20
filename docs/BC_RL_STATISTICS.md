# What is confirmatory in the BC→RL matrix

The 20-run matrix was executed before this note. The labels below do not change those numbers. They stop a 48-cell scan from being read as 48 independent discoveries.

## Primary comparison, original cyclic RL

Route Transformer versus its own behavior-cloning checkpoint, official n=10, construction only (P0), five seeds, one shared synthetic training set.

Paired difference (RL − BC): −0.091 h, about 0.4% of the behavior-cloning mean (22.379 h). Student-t 95% interval on the five seed differences: [−0.336, +0.154]. The interval includes zero. Under hybrid insertion the point estimate changes sign (+0.038 h, RL better on 1/5 seeds).

A second named comparison, same slice, is RT + DeepSets: +0.217 h, about 1.0% of its behavior-cloning mean (22.065 h), interval [−0.241, +0.676].

## Exploratory comparisons

The remaining cells in `results/tables/bc_rl_full_v1_paired.csv` vary model, size, and protocol. Forty-seven of 48 intervals include zero. The one that does not is Fleet DeepSets, n=10, hybrid insertion: +0.192 h, interval [+0.022, +0.362], RL better on 0/5 seeds. That cell was not predeclared. With 48 intervals, a single exclusion of zero is not treated as a confirmed effect. A Bonferroni reference level for 48 tests is 0.05/48.

## What the five seeds do and do not vary

Every seed uses the same 500 training instances and the same 100 validation instances (`data_seed=1000`). The seed changes initialization and sampling. The intervals do not include uncertainty from drawing a new training set. Official instances within a size share layouts, so instance-level rows are not independent replicates. The reported unit is the seed mean.

## Corrected RL sampler

The cyclic index `(step - 1 + b) mod 500` at 300 updates and batch size 2 visits 301 training instances. The shuffled sampler visits all 500 before repeating. The follow-up runs load the original `bc_best.pt` files and write to `results/raw/bc_rl_shuffle_v1`. The hypothesis frozen in that folder's `frozen_protocol.json`, before those runs, is that shuffled RL lowers synthetic validation duration relative to the original RL checkpoint.

That hypothesis is not supported. Paired difference (shuffled RL − original RL), in validation hours:

| Model | Mean difference | 95% interval | Seeds where shuffled is lower |
|---|---:|---|---:|
| Route Transformer | +0.035 | [−0.046, +0.116] | 1/5 |
| RT + DeepSets | +0.044 | [−0.033, +0.121] | 2/5 |

Positive means the shuffled checkpoint has the higher (worse) validation duration. Both intervals include zero. Against the behavior-cloning checkpoint, shuffled RL is lower on 1/5 Route Transformer seeds and 0/5 RT + DeepSets seeds. The RT + DeepSets difference versus behavior cloning is +0.114 h, interval [+0.011, +0.218]. That comparison was not the frozen hypothesis; it says the corrected RL checkpoint is worse than the checkpoint it started from, not that a new benefit was found. Official evaluation of these checkpoints uses the existing P0/P1/P2 protocol and is not used to accept or reject the sampler. Rows: `results/raw/bc_rl_shuffle_v1/validation_comparison.csv`.
