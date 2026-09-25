# Statistical gate

## What is being estimated?

For the same n examples, let `b_i` and `c_i` indicate correct baseline/candidate classifications. Define `d_i = c_i - b_i` in `{-1,0,+1}`. The observed accuracy difference is `mean(d)`.

The target is the accuracy difference under an example distribution represented by this benchmark, conditional on fixed model/configuration inference. This requires representative, approximately independent examples. A confidence interval does not establish robustness to a distribution shift or unseen classes of failure.

## Paired percentile bootstrap

The default is 10,000 seeded empirical resamples and the 2.5th/97.5th percentiles of their means. For this specific statistic, resampling the counts of `{-1,0,+1}` from a multinomial distribution is mathematically equivalent to resampling example indices. It reduces memory from `O(B*n)` to `O(B)`. It cannot be reused unchanged for F1.

The same ordered predictions, seed, resample count, code and NumPy version reproduce the interval. Swapping models negates the observed delta; finite Monte Carlo intervals need not mirror exactly under the same seed because category ordering changes the RNG draws. Tests check boundedness, reproducibility, hand-calculated outcomes and reversal within Monte Carlo tolerance.

## Verdict and margin

For interval `[L,U]` and preselected allowed loss `m > 0`:

- PASS if `L > -m`.
- FAIL if `U < -m`.
- INCONCLUSIVE otherwise, including equality at either threshold.

The default `m=0.01` is one accuracy point. The interval is two-sided 95%, which makes this a conservative non-inferiority decision rather than a conventional one-sided 95% test. The margin is a policy choice that should be selected before examining results. Changing it after viewing the data is an explicit risk decision, not fresh evidence.

A pass with the entire interval below zero is warned as statistically worse but within the accepted margin. Inconclusive has a separate CLI exit code and is never silently treated as pass.

## McNemar cross-check

Count baseline-right/candidate-wrong regressions `r` and the reverse fixes `f`. Conditional on `r+f`, an exact two-sided binomial test with null probability 0.5 tests zero paired difference. With no disagreements, report p=1.

This is another method on the **same evidence**, not an independent replication. It tests zero difference, not the non-inferiority margin. Disagreement with whether the bootstrap excludes zero produces an interpretation warning; it does not replace the gate.

## Precision and sparse cases

The plug-in standard error is `sqrt((discordant_rate - delta²)/n)`. With a 3% discordant rate near zero delta and n=872, the 95% half-width is roughly 1.15 accuracy points. A one-point margin can therefore yield an honest inconclusive verdict.

Percentile coverage is not universally 95%. For a true improvement of only 0.1%, an 872-example sample has probability `(1-0.001)^872 ≈ 0.418` of observing no improvements. Its empirical bootstrap is then degenerate at zero. Identical observed predictions legitimately produce `[0,0]`, but this is not proof of population equivalence.

Fewer than 20 discordant pairs produces a conservative interpretation warning. This is a transparent heuristic, not a theorem guaranteeing validity above the threshold. The project deliberately records a sparse-regime coverage failure rather than hiding it with BCa or a broad assertion. BCa alone does not solve missing support in a degenerate empirical sample.

Coverage tests use 500 synthetic datasets per moderate regime with n=872, balanced and asymmetric disagreements. They also demonstrate undercoverage for very sparse improvements. Wide regression-test tolerances accommodate Monte Carlo error; a passing test is not a universal calibration certificate. [Coverage measurements](results/coverage.json) report the specific seeded simulation settings and outcomes.

## Slices and probabilities

Long inputs (>25 whitespace-separated words), negation and contrast are deterministic, overlapping diagnostic tags. Only slices with at least 50 examples are shown. Their uncorrected intervals never change the overall verdict; they should not be interpreted as multiple confirmed discoveries.

`near_boundary` means either stored positive-class probability is within 0.01 of 0.5. It identifies sensitivity to small probability changes. It neither proves nondeterminism nor excuses a regression. A repeated fixed 100-example probe separately checks predicted-label stability, without proving determinism over every possible input or batch grouping.

Generative predictions have null probability and stored raw answers. Exact lowercase positive/negative is required after whitespace stripping; invalid output is -1 and always incorrect. Invalid examples remain in accuracy, F1, paired bootstrap and slice denominators. Boundary diagnostics use only available classifier probabilities. The generative stability probe repeats ten examples. See [the classifier/LLM protocol](llm-experiment.md).

## Measured result

The int8 candidate has 16 regressions and 9 fixes: `delta=-7/872≈-0.00803`. Its interval `[-0.01950,0.00344]` crosses `-0.01`, so the verdict is INCONCLUSIVE. The truncation candidate has 104 regressions and 18 fixes: `delta=-86/872≈-0.09862`, with interval `[-0.12271,-0.07569]`, entirely beyond the margin, so it fails.

These outcomes support the product decision on this pinned benchmark. They do not show how a model behaves on production traffic.
