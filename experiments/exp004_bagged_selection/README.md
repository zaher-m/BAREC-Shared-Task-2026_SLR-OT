# E4. Bagged selection, and the collapsed member hiding underneath it

**When** Aug 2 · **Verdict** bagging's gain was mostly a collapsed member; dropping it is the real fix

Question. sub02 scored 85.2 on blind against a CV estimate of 86.623, a -1.42 shift that
matches v5's (86.398 -> 85.00) and thylinao's (86.30 -> 84.80). If that shift is a constant
then public-test CV is a calibrated predictor of blind, and beating thylinao's 85.3 needs CV
above about 86.75. Two things to spend: the pool had grown 45 -> 67, and exp003's 0.75 gap
between in-sample and CV was pure selection variance, which bagging is the standard cure for.

## Results (document-grouped 5-fold CV on TEST, everything refit per fold)

On the 67-member pool (before the audit):

| method | CV QWK |
|---|---|
| bagged greedy, model_frac 0.35 | 86.839 |
| bagged greedy, model_frac 0.50 | 86.811 |
| bagged greedy 0.50 + row subsampling 0.8 | 86.808 |
| greedy (45-member subset) | 86.785 |
| greedy | 86.715 |
| greedy + bagged thresholds | 86.709 |
| top-30 uniform | 86.444 |
| top-20 uniform | 86.275 |

After the audit removed one member (exp005, 66-member pool):

| method | CV QWK |
|---|---|
| greedy | 86.792 |
| bagged greedy 0.50 | 86.796 |
| z-scored + greedy | 86.692 |

## Read

1. A collapsed member was sitting in the pool. `arabertv2_large_reg_ad` had score
   sd = 0.000 on both test and blind and a solo QWK of 41.0: training diverged and it
   emitted a constant. Dropping it is worth +0.077 on its own, and it had been masking
   the pool gain. With it in, 67 members scored below 45; with it out, 66 members beat
   45. Every member is now audited for degeneracy (`members.DEGENERATE`).
2. Most of what bagging appeared to buy was really that. Bagging is +0.096 on the
   dirty pool but only +0.004 on the clean one, so averaging over model subsets was
   diluting the dead column rather than fixing genuine selection variance. Both effects
   were real, but they were largely the same effect, and reporting them as additive
   would have overstated the system by about 0.1.
3. Bagged thresholds do nothing (86.709 vs 86.715). The 18 boundaries are not where
   the overfitting lives; the weights are.
4. Uniform top-k is far behind (-0.35 to -0.52). Weighting earns its keep.
5. Normalisation hurts (exp005): z-scoring costs 0.10 (86.792 -> 86.692). Member score
   spreads vary only 1.21x, so there is nothing to correct and the transform just discards
   scale information the thresholds were using.

Later note on that last point: exp005's own file also has rank-normalisation with bagged
greedy at 86.876, which is +0.08 over raw. That is inside the noise band exp016 later
established for this CV, and it was never followed up, so it does not change the verdict. It
is recorded here rather than left out of the file.

## Method note

These runs are only affordable because of `thresholds.FastQWKThresholds`: parameterising
the 18 boundaries as cut indices into the sorted score vector makes a single-position
move an O(1) update to QWK's numerator and denominator, so a full coordinate-ascent sweep
is O(n) instead of O(n * candidates). 71-82x faster, and it finds better optima than the
reference optimiser, since it searches every cut position rather than a 0.02 grid: worth
+0.16 by itself on the same ensemble. This is where `fit_thresholds` switched over to it,
which is why exp001 to exp003 have to ask for the grid explicitly to reproduce.

Later note: the better optima do not transfer. exp009 found the grid was acting as a
regulariser and the exact optimiser costs 0.3 QWK on blind.

## Reproducing this exactly

`python experiments/exp004_bagged_selection/run.py`

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp004: try to claw back the 0.75 QWK of selection overfit on the bigger pool | scratch root |
