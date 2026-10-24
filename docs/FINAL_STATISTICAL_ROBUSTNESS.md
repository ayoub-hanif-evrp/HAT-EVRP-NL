# Statistical robustness

The experimental unit for BC versus RL is the training seed. Five seeds share one synthetic training set (`data_seed` 1000). They measure initialization and sampling noise, not a new draw of the training data.

A benchmark instance measures one point inside a fixed 20-instance set. A sample trial estimates one draw from one policy on one instance. The mean of four trials is that estimate. It is not 400 independent training runs, and it is not a route whose duration equals the mean.

## Central comparisons

Route Transformer, official instances, sampled mean, RL minus BC. Source: `results/tables/final/central_seed_differences.csv` and `results/tables/final/statistical_sensitivity.csv`.

| n | Seed differences (h) | Mean | Percent | SD | Student-t 95% interval | t p | Sign p | Wilcoxon p |
|---:|---|---:|---:|---:|---|---:|---:|---:|
| 10 | −0.574, −0.929, −0.900, −0.545, −0.622 | −0.714 | −2.72% | 0.185 | [−0.944, −0.484] | 0.00100 | 0.0625 | 0.0625 |
| 20 | −0.183, −0.927, −0.683, −0.761, −0.274 | −0.566 | −1.33% | 0.322 | [−0.965, −0.166] | 0.0171 | 0.0625 | 0.0625 |

All ten differences are negative. There are no zeros and no ties in absolute value.

Holm adjustment for the two central tests:

| Test | n=10 raw | n=20 raw | n=10 Holm | n=20 Holm |
|---|---:|---:|---:|---:|
| Paired Student-t | 0.00100 | 0.0171 | 0.00200 | 0.0171 |
| Exact two-sided sign | 0.0625 | 0.0625 | 0.125 | 0.125 |
| Exact Wilcoxon signed rank | 0.0625 | 0.0625 | 0.125 | 0.125 |

The sign test is the two-sided binomial probability of five differences of one sign. That probability is \(2\times(1/2)^5=0.0625\). It ignores magnitudes. The Wilcoxon test ranks the absolute differences and assigns signs. With every sign the same, the positive rank sum is 0, and the exact two-sided probability over the 32 equally likely sign assignments is also 0.0625. Wilcoxon additionally assumes that the paired differences are symmetric about zero under the null. The t-test assumes that the five differences are normal. Five points cannot check normality. No one-sided test was substituted after seeing the signs.

The t-tests remain below 0.05 after Holm correction. The sign and Wilcoxon tests do not. The manuscript therefore reports a consistent direction and a normal-theory interval, and it does not say that five out of five establishes two-sided 5% significance.

## Exploratory analyses

RT + DeepSets, best-of-four, other architectures, n=40, and the charging-component splits are in the same CSV with status `exploratory`. They were inspected with the central cells. They are not a second Holm family in the paper.

A cluster bootstrap resamples the five seeds and, inside a seed, the 20 instance-level sampled means. The four draws stay inside the instance mean. For the Route Transformer, 4,000 replicates give percentile intervals [−0.978, −0.439] at n=10 and [−0.998, −0.163] at n=20 (`results/tables/final/hierarchical_bootstrap.csv`). Those intervals use the magnitudes. Five clusters do not make the bootstrap a substitute for the sign test.

Within one official instance, the standard deviation of the four Route Transformer draws is about 3.07 h at n=10 and 3.67 h at n=20 (`results/tables/final/sample_expectation_vs_selection.csv`). That is Monte Carlo noise of one policy, not training-seed uncertainty.
