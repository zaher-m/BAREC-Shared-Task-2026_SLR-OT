# E13. Uncertainty shrinkage

**When** Aug 2 · **Verdict** dead: +0.02 inconsistent; the far tail is not where members disagree

## Question

Same metric asymmetry as exp012, tail theory: MAE equal while Acc19 and +-1 are ahead means the
error mass sits further out. Shrink each sentence's score toward the mean in proportion to
member disagreement?

## Method

`s' = m + (s - m) * t2 / (t2 + lam * var_i)` with `var_i` the across-member variance on row i,
swept over lam on both transfer directions (dev to test and test to dev).

## Recorded outcome

Best +0.02 and not consistent across directions, so noise. The diagnostic that closes it: the
err>=6 rate barely moves (0.89% -> 0.88%) across the whole lam range. The far tail is not where
members disagree; the whole pool is confidently wrong there, and no reweighting of the same
pool can fix that.

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp013: shrink the confident-but-wrong tail, per sentence | scratch root |
