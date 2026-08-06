# E16. Pruning the pool (negative), and a correction to the saturation story

**When** Aug 3 · **Verdict** pruning is strictly worse; the -0.08 was CV noise

Question. Pool-size results had turned over: 10->39 was +0.19, 39->66 was +0.007, 70->71 was
-0.08. If that is selection variance from too many correlated columns on 7,286 calibration
rows, the remedy would be fewer, better members. exp004 tested top-k with uniform weights and
it lost badly, but that changed two things at once, pruning and discarding the weighting. This
isolates pruning: rank members by solo calibrated QWK inside each CV fold, keep the top N, then
run the same capped greedy and prior-shrunk thresholds over the survivors.

Ranking happens inside the fold, on training rows only. Ranking on the full set first would
leak the pruning decision into the rows it is scored on.

## Result: pruning is clearly worse

| top N | CV QWK (cap30) | CV QWK (clean) |
|---|---|---|
| 10 | 86.388 | 86.114 |
| 16 | 86.433 | 86.310 |
| 24 | 86.355 | 86.361 |
| 36 | 86.513 | 86.542 |
| all 73 | 86.717 | 86.569 |

Monotone in the wrong direction for the hypothesis: every pruned pool loses and the full pool
wins. Breadth is still doing real work even at 73 members.

## What this corrects

The -0.08 we attributed to "the second silver member hurting" was CV noise, not saturation.
Three further silver members moved the full-pool number from 86.713 to 86.717, which is flat
rather than negative. The honest statement is that pool breadth has plateaued, new members
neither help nor hurt, and the plateau caps gains without being evidence that the pool is too
large. Removing members is strictly worse.

So "adding members is finished as a strategy" was right for the wrong reason. Adding stopped
paying, it did not start costing.

Note also that every pruned pool has higher Acc19 (40-41 vs 39.1) and lower QWK, the same
accuracy/QWK trade-off seen in exp002's prior-matched thresholds and exp009's shrinkage sweep.
Acc19 is not a proxy for the ranked metric, in either direction.

## Consequence

With combination, thresholds, post-processing and now pool composition all measured, the only
remaining path is a single substantially stronger member, one good enough to take heavy weight
and lift a plateaued combination rather than join it. That is `arabertv2_large_emd_slv`
(bert-large plus the silver->gold recipe).

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp016: if the pool has stopped improving, is pruning better than growing? | scratch root |
