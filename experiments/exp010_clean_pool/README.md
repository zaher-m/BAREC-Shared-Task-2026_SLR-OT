# E10. The contaminated-member problem, and what to do about it

**When** Aug 2 · **Verdict** contamination measured at 1.06 QWK; clean pool alone tops out ~86.52

## The problem

Every `_ad` member early-stopped on the BAREC public test split, the same rows used for
ensemble selection and for fitting the 18 thresholds. Measured inflation, from the training
logs (best epoch minus mean epoch): 1.06 QWK, n=18.

So every estimate computed on TEST overstates an `_ad`-heavy ensemble, in proportion to how
much weight it puts on them, which matches the offsets already on the board:

| submission | weight on `_ad` | CV estimate | blind | offset |
|---|---|---|---|---|
| sub01 (v5) | 0% | 86.398 (test, honest) | 85.0 | -1.40 |
| sub02 | ~30% | 86.623 | 85.2 | -1.42 |
| sub03 | majority | 86.796 | 84.9 | -1.90 |

exp003's "+0.180 from `_ad` members" and exp006's "+0.382" were both measuring this
inflation rather than skill. sub03 was built on those numbers, which is why it lost.

Arithmetic that isolates it: on clean dev->test transfer (exp009) sub03's combiner and
threshold changes were +0.35 good, blind says sub03 was 0.3 worse, so the pool change is
worth about -0.65.

## exp010: the clean pool alone

39 members trained on the BAREC train split only. They early-stopped on DEV, so DEV is
contaminated for them but TEST is untouched by anything, which makes it honest to fit and
calibrate on.

| combiner | threshold rule | CV QWK | Acc19 |
|---|---|---|---|
| greedy | free (sub03's rule) | 86.410 | 36.62 |
| greedy | prior-0.25 | 86.452 | 38.25 |
| greedy | prior-0.50 | 86.482 | 39.67 |
| greedy | prior-0.65 | 86.523 | 40.42 |
| bagged | free | 86.458 | 35.75 |
| bagged | prior-0.25 | 86.513 | 37.39 |

Prior shrinkage helps monotonically and lifts Acc19 by about 4 points, which attacks the
failure that sank sub03 directly. But the clean pool tops out around 86.52, implying blind
~85.1, which does not reach thylinao's 85.3 on its own. Purity is not sufficient.

## exp011: cap rather than exclude

The `_ad` members are not bad models, they are trained on 13% more data. They just cannot be
trusted to select their own weights. And the only uncontaminated evidence available, the
three blind results above, shows a clean interior optimum near 30%.

`combiners.CappedGreedy` therefore runs ordinary greedy with a hard ceiling on the total
weight the flagged members may hold. The cap comes from the blind measurements, not from any
number computed on the contaminated split. No estimator available here can see the inflation,
so trying to re-derive the cap on that split would be circular.

Read exp011's CV column with care: it is computed on the contaminated split, so it rises with
the cap by construction. That rise is an artefact, not evidence the cap should be higher. The
trustworthy columns are Acc19 and the minimum gap between cuts.

## Shipped as sub04

66-member pool, `_ad` weight capped at 30% (lands at 28.6%), prior-0.50 thresholds.

- minimum gap between adjacent cuts 0.505 (sub03: collapsed)
- only empty level is 19, same as sub02. sub03 had 3 and 19
- level 3 restored to 1.46% (sub03: 0.00%, sub02: 2.24%)
- 84.8% exact agreement with sub02, the best submission so far, so this is its closest
  relative of the three

Against sub02 it changes three things, each independently supported: a larger clean pool,
prior-shrunk thresholds (+0.17 on honest dev->test transfer), and the degenerate member
removed. Expectation ~85.4, and unlike sub03 the estimate is not resting on a contaminated
comparison.

Later note: it scored 85.2, tying sub02. Capping was superseded by dropping the `_ad`
members altogether (exp017), which is what sub07 did.

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp010: rebuild on the uncontaminated pool, with the threshold rule that transfers | scratch root |
