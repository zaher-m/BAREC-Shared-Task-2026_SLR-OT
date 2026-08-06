# E3. Do the `_ad` members earn their place, and what gets submitted?

**When** Jul 27 · **Verdict** shipped sub02 (blind 85.2); its +0.180 for _ad later shown to be inflation

Question. The 6 `_ad` members were retrained on train+dev (62,155 sentences, +13% over the
54,845 the strict members saw). Extra data is only worth having if it survives an honest
estimate, and it cannot be measured the usual way: DEV is inside their training set, so the 39
strict members and the 6 `_ad` members share no held-out set except TEST.

Protocol. Both pools go through the identical procedure, document-grouped 5-fold CV on TEST
with the greedy selection and the 18 thresholds refit inside every fold
(`combiners.cv_qwk_full`). That measures the procedure rather than one lucky fit, and it is the
only comparison where the `_ad` members' extra data can be credited without also crediting
their head start on DEV.

## Result

| pool | members | CV QWK on TEST |
|---|---|---|
| P39 (strict members only) | 39 | 86.443 |
| P45 (+ 6 `_ad`) | 45 | 86.623 |

The 6 train+dev members are worth +0.180 QWK, independently of and roughly equal to the +0.19
that exp002 got from widening the pool. The two effects stack, which is why the next campaign
(`configs/campaigns/02_large_and_seeds.yaml`) buys both at once: new backbones and the `_ad`
protocol.

Reference point: P39 fit on DEV and scored on TEST gives 86.590. That is a genuinely held-out
number and it sits above P39's own CV estimate (86.443), which is the expected direction, since
the CV folds fit on 4/5 of the calibration data and so under-estimate slightly.

## Shipped system (sub02)

45-member pool, greedy multiset with replacement, weights and thresholds both fit on TEST.
16 members take non-zero weight, top four at 10% each: `arabertv2_emd_d3`, `arabertv2_large_corn`,
`araelectra_corn_ad`, `aramodern_reg_ad`.

In-sample TEST QWK is 87.369, but that is the number to ignore: the gap to the CV estimate
(86.623) is 0.75 QWK of selection overfitting, which is what fitting 45 weights and 18
thresholds on 7,286 rows costs. The honest expectation is 86.62 on a test-like set.

Expected blind score ~ 85.2. The v5 ensemble scored 86.398 on TEST and 85.00 on blind, a
-1.40 shift that every team on the leaderboard shows (thylinao: 86.30 -> 84.80). Applying the same
shift to 86.62 lands at 85.2.

## Sanity checks

- 75.7% exact agreement with sub01, QWK 98.56 between the two prediction files. The systems are
  close relatives, as they should be, and the disagreements are almost all +-1.
- Blind score distribution (mean 10.309, sd 2.738) sits slightly below the TEST fit (10.451, 2.880),
  consistent with the blind set being marginally easier-skewed, not with a scoring bug.
- Predicted blind label histogram tracks the TEST gold histogram in shape. The visible gaps
  (level 5: 1.9% vs 5.2%; level 9: 9.3% vs 2.6%) are real distribution differences between the two
  sets, not clipping: predictions span 1..18 with no pile-up at the boundaries.

## Reproducing it

Like exp002, this needs two things pinned to come out the same: the 45 members in `pool.txt`,
and the 0.02-grid cuts and greedy objective (`CUTS`, `GREEDY`), which is what the code did
before the exact optimiser existed. Re-running does not overwrite `sub02`'s uploaded zip, it
compares against it.

With those pinned, the shipped system reproduces exactly: the same 16 weighted members, the same
18 cuts, the same in-sample 87.369, and the same dev-fit reference 86.590. The two
`cv_qwk_full` numbers do not: P39 comes back 86.5064 against the recorded 86.4430, and P45
86.6549 against 86.6233, so the `_ad` gain is +0.148 rather than +0.180.

Something in the CV wrapper changed between this run and the final code, and the record does not
say what. Ten variants were measured trying to find it, and P39 lands at 86.4235 (4 folds),
86.4543 (exact cuts), 86.4656 (0.05 grid), 86.5064 (0.02 grid, the current code), 86.5168
(greedy r30), 86.5209 (coarse 0.10 grid) and 86.3052 (fold seed 42). Averaging per fold instead
of pooling gives ~87.9, so that is not it either. The recorded number sits inside that band,
which is worth knowing on its own: the +0.15 to +0.18 conclusion is stable across every one of
those choices, and it was inflation in all of them (exp010).

## Reproducing this exactly

`python experiments/exp003_production_ensemble/run.py`. Uses the grid-era cuts and greedy objective (`CUTS`, `GREEDY`). The shipped ensemble, its 16 weights, its 18 cuts, the in-sample 87.369 and the dev-fit 86.590 reproduce exactly; the two `cv_qwk_full` numbers come back 0.03-0.06 high, see the reproduction note at the end of this page. The sub02 zip is compared against, never overwritten.

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp003: pick the ensemble to submit and write its blind prediction | scratch root |
