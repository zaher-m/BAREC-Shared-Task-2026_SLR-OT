# E9. Why sub03 lost, and which threshold rule transfers

**When** Aug 2 · **Verdict** free cuts overfit; prior-shrunk cuts transfer (+0.17 dev to test)

What happened. sub03 shipped on a CV estimate of 86.796 (implied blind 85.38) and scored
84.9, below sub02's 85.2. Offset -1.90 instead of the -1.42 seen twice before.

The signature. Acc19 fell 38.9 -> 33.2 (-5.7) while +-1 accuracy rose 72.8 -> 73.1.
Predictions landed one level off nearly everywhere. That is misplaced decision boundaries,
not a worse score vector. Comparing the shipped files confirms it:

| level | sub02 | sub03 |
|---|---|---|
| 3 | 2.24% | 0.00% |
| 6 | 1.63% | 5.52% |
| 7 | 5.47% | 1.15% |

Two adjacent cuts collapsed onto each other and level 3 went empty. The pipeline is not at
fault: sub02 reproduces 100% from the cached blind scores.

## Cause 1: the exact threshold optimiser overfits

Parameterising cuts by position in the sorted score array lets a boundary sit anywhere
between two adjacent scores, so 18 boundaries can chase individual calibration points.
The old 0.02-value grid was coarse enough to act as an accidental regulariser. Our
earlier claim that the exact optimiser was "worth +0.16" was wrong twice over: it was
inferred by differencing two runs that also differed in pool and combiner, and "finds
better optima on the calibration set" is the problem, not the benefit.

## Cause 2: within-test CV is the wrong instrument

It resamples documents from one split, so it measures variance within a document sample,
never degradation onto a different one. It ranked sub03 above sub02.
DEV -> TEST transfer is the right shape: two different document samples, labels on
both sides, same structure as calibrate-on-test -> predict-blind.

## Results (fit on dev, applied to test, 39 clean members)

| combiner | rule | TEST QWK | Acc19 | min gap between cuts |
|---|---|---|---|---|
| greedy | ref-grid (sub02's) | 86.236 | 37.77 | 0.020 |
| greedy | exact (sub03's) | 86.311 | 37.06 | 0.041 |
| greedy | bagged | 86.427 | 37.70 | 0.281 |
| greedy | prior-0.50 | 86.410 | 39.95 | 0.275 |
| bagged greedy | exact | 86.581 | 35.79 | 0.287 |
| bagged greedy | prior-0.50 | 86.602 | 39.34 | 0.272 |

Shrinking the cuts halfway to the label prior is worth +0.17 QWK and +2.2 Acc19 over a
free fit, and raises the minimum gap between adjacent cuts from 0.04 to 0.28, which is what
structurally prevents an empty level. k=0.5 is the optimum. k=1.0 (pure prior) maximises
Acc19 at 41.1 but gives QWK back.

## The residual puzzle, which becomes exp010

dev->test still rates sub03's recipe (bagged + exact, 86.581) above sub02's (greedy +
ref-grid, 86.236) by +0.35, while blind says it is 0.3 worse. Threshold overfitting alone
does not close a 0.65 gap. The missing term is the pool: dev->test can only run on the 39
clean members, so it never tested the thing sub03 changed most, which was adding 27 `_ad`
members that early-stopped on the calibration set. See exp010.

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp009: why sub03 lost, and which threshold rule transfers | scratch root |
