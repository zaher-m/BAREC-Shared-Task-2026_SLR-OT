# E12. Chasing the metric asymmetry (both negative)

**When** Aug 2 · **Verdict** dead: gamma=1.00 already optimal, the threshold fit transfers

## The observation

sub04 vs thylinao on the same blind set:

| | Acc19 | +-1 | Acc7 | Acc5 | Acc3 | MAE | QWK |
|---|---|---|---|---|---|---|---|
| sub04 | 39.4 | 72.6 | 60.8 | 68.6 | 75.9 | 1.1 | 85.2 |
| thylinao | 37.7 | 70.8 | 60.5 | 66.9 | 74.6 | 1.1 | 85.3 |

Five of six metrics go to sub04 and it still loses the ranked one. 1.7 points of Acc19 on
8,077 rows is far outside standard error, so this is structural rather than noise.

## Hypothesis 1: predicted spread (exp012)

QWK is `1 - sum(W*O) / sum(W*E)` with `E_ij = n_i^true * n_j^pred / N`. Holding the true
marginal fixed, the denominator depends only on the predicted marginal and grows as that
marginal spreads away from centre. Two systems with identical errors therefore score
differently if one predicts extremes more often. sub04's marginal is concentrated: 46% of
blind in the top three levels, level 19 never predicted. And the ensemble score is compressed
on blind (sd 2.735 vs 2.883 on test), so thresholds fitted on test produce an even narrower
marginal there.

Swept a spread factor gamma about the score mean, both transfer directions, free and
prior-0.50 thresholds:

| gamma | 0.90 | 0.95 | 1.00 | 1.05 | 1.10 | 1.20 | 1.30 |
|---|---|---|---|---|---|---|---|
| mean dQWK (free) | -0.71 | -0.27 | 0.00 | -0.09 | -0.42 | -1.55 | -3.26 |
| mean dQWK (prior-0.50) | -0.51 | -0.16 | 0.00 | -0.01 | -0.38 | -1.63 | -3.26 |

gamma = 1.00 is exactly optimal and falls away symmetrically. The threshold optimiser already
finds the right spread and it transfers. Dead, and it also closes exp007's loose end: that
experiment tested only the single matched gamma, and the full curve confirms the peak is at 1.

## Hypothesis 2: a heavy far tail (exp013)

MAE is equal while sub04 wins at both 0 and <=1 error. Arithmetically the error mass has to
sit further out, so sub04 has a heavier far tail and QWK punishes it quadratically. Under
squared loss the optimal prediction for an uncertain case shrinks toward the mean, so shrink
per sentence by across-member variance:
`s' = m + (s - m) * t2 / (t2 + lam * var_i)`, with `t2` the variance of the ensemble score.

| lam | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 | 0.80 |
|---|---|---|---|---|---|---|
| mean dQWK | +0.018 | -0.003 | +0.020 | +0.015 | +0.010 | +0.005 |
| consistent across directions | yes | yes | no | no | no | no |

Best is +0.02 and not consistent, so noise. The diagnostic detail that matters: the err>=6
rate barely moves (0.89% -> 0.88% across the whole lam range). The far tail is not where the
members disagree, so member variance is the wrong signal for finding it. Those sentences are
ones the whole pool is confidently wrong about, which no reweighting of the same pool can fix.

## Consequence

Together with exp001, exp005, exp007 and the BAREC-10M metadata check (residual R2 0.0017),
every post-processing avenue is measured and closed. The member pool is the constraint, which
is what moved the remaining effort to silver-label pretraining.

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp012: five of six metrics ahead of thylinao, and the ranked one behind | scratch root |
