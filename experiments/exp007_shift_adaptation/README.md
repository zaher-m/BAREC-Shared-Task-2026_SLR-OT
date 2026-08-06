# E7. Is the public-test -> blind drop a fixable score shift?

**When** Aug 2 · **Verdict** dead: aligning the score distribution costs QWK in both directions

Question. Every team loses ~1.4 QWK going from the public test set to blind (this system
86.40->85.00 and 86.62->85.2, thylinao 86.30->84.80). exp003 found the ensemble's blind
scores are compressed relative to the calibration set (sd 2.738 vs 2.880). Thresholds fit on
the wider distribution then cut the narrower one too far out, under-predicting the extreme
levels, and QWK punishes that quadratically. If the compression is an artefact of members
being less certain off-distribution, rescaling recovers free QWK. If it is real, rescaling
invents variance that is not there.

How to settle it without touching blind. DEV and TEST are both genuinely held out
for the 39 train-only members, and they are two different document samples: the same kind
of shift, with labels on both sides. Fit on one, predict the other, try to align.

## Result: alignment hurts, in both directions

| direction | none | affine | spread-only | quantile |
|---|---|---|---|---|
| dev -> test (sd ratio 0.961) | 86.311 | 85.845 | 86.270 | 85.622 |
| test -> dev (sd ratio 1.040) | 85.228 | 84.368 | 84.993 | 84.350 |
| mean dQWK | - | -0.663 | -0.139 | -0.784 |

Every variant loses, in both directions, and the fuller the correction the worse it
gets: quantile mapping (matches the whole shape) -0.78, affine (mean+spread) -0.66,
spread-only (the mildest) -0.14.

## Read

The compression is real signal, not miscalibration. A held-out document sample
genuinely contains a different mix of readability levels, and the ensemble is right to
report a narrower spread on it. Forcing the predicted distribution back to the
calibration set's shape overwrites a correct prediction with a wrong prior.

The direction of the damage confirms the mechanism. `test -> dev` has sd ratio 1.04, so
alignment shrinks the target, and it costs 0.86 QWK and collapses Acc19 from 35.1 to 25.4.
Alignment is not mildly unhelpful, it destroys the ordering near the boundaries.

Consequence for the plan. The -1.4 blind gap is intrinsic difficulty, not something
post-processing can recover. The way to a better blind score is a better system, and
public-test CV minus 1.42 stays the honest predictor of where a submission will land. This
also closes the door on any transductive or test-time-adaptation variant of the same idea.

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp007: is the -1.4 drop from public test to blind partly a fixable score shift? | scratch root |
