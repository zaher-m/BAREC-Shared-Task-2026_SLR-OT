# E6. Is the calibration set, not the pool, the cap?

**When** Aug 2 · **Verdict** 2x calibration rows worth +0.11; the pool is worth more

Question. exp003 measured 0.75 QWK between in-sample (87.369) and CV (86.623), the cost
of fitting weights and 18 thresholds on 7,286 rows. exp004 then found the larger pool
scoring worse under plain greedy, which is the same disease getting worse as the library
grows. Selection variance falls with calibration size, and there is a second held-out set
sitting unused: DEV is genuinely held out for the 39 train-only members, though not for the
`_ad` members, which trained on it. So it is a straight trade, 2x the calibration rows or
0.59x the pool.

Fair comparison. Both systems are scored on held-out folds of TEST with the same
document-grouped folds. The only difference is what each may fit on: P39 gets all of DEV
plus four-fifths of TEST (14,596 rows), P66 gets only those four-fifths (7,286).

## Result

| system | pool | calibration rows | CV QWK |
|---|---|---|---|
| P66 / test only | 66 | 7,286 | 86.792 |
| P39 / dev+test | 39 | 14,596 | 86.524 |
| P39 / test only | 39 | 7,286 | 86.410 |

## Read

Doubling the calibration set is worth +0.114 (86.410 -> 86.524). Real, and a useful
constant to carry: it prices selection variance directly, and says roughly what any
future increase in calibration data would buy.

But the 27 `_ad` members are worth +0.382 (86.410 -> 86.792), more than three times as
much. The pool wins decisively and the production design stays P66 / test-only.

The tempting hybrid, retraining strong configs while holding out DEV instead of TEST so the
pool is honest on dev+test, does not actually help. It would swap the `_ad` members for an
equal number of new ones rather than adding any, buying the +0.11 while giving back the
+0.38. Not worth GPU time.

Standing implication. Calibration size is a genuine but second-order lever here. If a future
round wants it, the way to get it is grouped k-fold over all 69,441 labelled sentences so
that every member has honest out-of-fold scores everywhere, not dropping members to free up
a held-out split.

Later note: the +0.382 for the `_ad` members is inflation, not skill. They early-stopped on
TEST, which is the split this is scored on. See exp010.

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp006: is the calibration set the cap, rather than the pool? | scratch root |
