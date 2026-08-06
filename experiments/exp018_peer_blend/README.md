# E18. Reverse-engineering the ceiling: is there information outside the pool?

**When** Aug 3 - Aug 5 · **Verdict** sents_RL is not an answer key; peer blend refuted by measurement; retrieval worth +0.05, post-deadline

Run after the Testing Phase closed, so nothing here can be scored.
The question was whether anything in the available public resources carries readability
information that the 76-member pool does not, since twelve measured negatives had established
that no amount of re-combining the pool moves the number.

## The hypothesis worth testing: is `sents_RL` an answer key?

BAREC-10M ships a `sents_RL` field, a 19-level readability label, for ~552K sentences, and
57% of the blind sentences appear in that corpus. If those labels were gold, the Open Track
was a retrieval problem, not a modelling one. This was never checked, despite the corpus
having been used for silver pretraining in exp014.

It agrees with gold at QWK 97.96 / 95.0% exact on dev+test. That is either a copy of the
human label or a badly overfit model scoring its own training data.

### Test A: is it a memorising model? No.

| split | n | QWK | exact % | MAE |
|---|---|---|---|---|
| train | 63,848 | 98.280 | 94.20 | 0.134 |
| validation | 9,431 | 97.418 | 94.12 | 0.169 |
| test | 13,421 | 98.305 | 95.66 | 0.115 |

Agreement is flat across splits (train minus held-out: -0.69 points). A model trained on train
would lose tens of points on dev/test. So `sents_RL` is the human BAREC label copied through the
pipeline, with ~5% loss from sentence-splitting and de-duplication mismatches.

That is the trap. The 97.96 invites the conclusion that the field is gold everywhere, when what
it actually shows is that the field is gold where BAREC has a label.

### Test B: are the blind labels gold too? No.

Blind sentences carry no BAREC label, so the pipeline had nothing to copy and fell back to
prediction. Two independent confirmations:

1. Agreement with sub07 is too high. QWK(`sents_RL`, sub07) on the 4,274 covered blind rows is
   90.84, while sub07 truly scores 85.50. Truth cannot agree with a predictor better than the
   predictor's own accuracy. Two correlated models can.
2. Subset difficulty does not explain it. For the 90.84 to be agreement-with-truth, sub07 would
   have to be +5.34 better on the covered rows than overall. Fitting P(covered | length, score)
   on blind and reweighting the public test split toward the covered-like distribution gives
   -0.60, so the covered rows are marginally harder. Score spread is also flat across the two
   subsets (sd 2.745 covered vs 2.688 uncovered), which rules out the QWK-denominator
   artefact.

Verdict: no leak. Had there been one, exploiting it would have been label retrieval rather
than readability modelling, and worthless in the paper.

## What the refutation left behind

`sents_RL` on blind is a peer system's out-of-sample prediction, which is a resource this
project never had. All 76 members share training data and architecture family and their
residuals correlate at ~0.997, which is why exp002 (stacking), exp005 (normalisation), exp007
(alignment), exp012 (spread), exp013 (shrinkage) and exp004 (bagging) all measured flat or
negative. There was nothing left to recombine.

Measured relationship on the covered blind rows:

| quantity | value |
|---|---|
| coverage | 4,274 / 8,077 (52.9%) |
| pearson r(sub07, peer) | 0.9231 |
| D = E[(sub07 - peer)^2] / var(sub07 error) | 0.488 |
| implied error correlation rho | ~ 0.76 |

For comparison, the strongest pool member against the ensemble: D/v_o = 0.11, rho = 0.95. The
peer is about 4.5x more decorrelated from the ensemble than its own members are.

### A note on the three coverage numbers in this directory

They differ for understood reasons, and only the last is the one to use:

- 4,612 (`s01_measure.py`) matched against `silver_raw.parquet`, which excludes dev/test but
  keeps train. So it includes blind rows whose text matches a train sentence, and those carry
  the gold label rather than a prediction. Contaminated for this purpose.
- 4,274 (`s03_decisive.py`) excludes every gold-matched text, but its `key -> index` map keeps
  only the last of any duplicated blind text, so ~272 rows are silently dropped. An undercount,
  though it does not affect that script's conclusion, which is about a QWK ratio.
- 4,546 / 56.3% (`s07_build_final.py`) excludes gold-matched text and assigns the label to every
  blind row sharing it. This is the correct figure and the one the shipped file uses.

## The trap: why `sents_RL` must never be a pool member (`s09_trap.py`)

The obvious move is to add the column to the pool and let the combiner decide its worth. On the
calibration split, the BAREC public test set, that column is the gold label (QWK 98.31, 95.29%
exact). So:

| | in-sample TEST QWK | document-grouped 5-fold CV |
|---|---|---|
| clean pool, 48 members | 86.939 | 86.504 |
| + `sents_RL` as a 49th member | 98.365 | 98.317 |

Greedy hands it 92.0% of the weight. Both instruments, including the grouped CV that this
project spent the whole competition learning to trust, endorse building the entire system on a
column that is gold here and an ordinary prediction on blind.

No resampling scheme inside the calibration split can detect this, because every fold sees the
column as gold. Knowing where the column came from is the only thing that catches it. That is
why the peer enters `sub10` through a post-hoc blend whose weight comes from an external
variance argument and is never fitted, and it is the same lesson as the `_ad` members in
exp010/exp011, in a stronger form.

## Channel A: retrieval against public gold (validated, +0.094)

A blind sentence near-identical to one in BAREC train/validation/test has a publicly known
label. This is distance-~0 retrieval over public data, and nothing held out is touched.

Dry run on the public test split with a train+dev-only index, character 3-5 gram TF-IDF
cosine:

| gate | rows | % | retrieved label exact | base MAE there | dQWK hard | dQWK alpha=0.5 |
|---|---|---|---|---|---|---|
| >=0.99 | 366 | 5.02 | 98.6 | 0.538 | +0.012 | +0.068 |
| >=0.90 | 406 | 5.57 | 95.3 | 0.576 | +0.036 | +0.094 |
| >=0.80 | 506 | 6.94 | 88.7 | 0.704 | -0.056 | +0.065 |
| >=0.60 | 848 | 11.64 | 68.8 | 0.987 | -1.420 | -0.376 |

The gate is doing real work: ungated similarity-weighted 5-NN is flat to negative at every
setting tried (best +0.010). Precision collapses below 0.85 and takes the gain with it.

Strict verbatim matching found only 160 test rows (2.2%) against retrieval's 406.
Character n-grams catch punctuation and diacritic variants that exact normalisation misses.

## Channel B: peer blend, measured, and it does not work

### What the variance theory predicted

The peer's accuracy on blind is unmeasurable: there is no gold there, and on every split where
gold exists the peer is the gold. So it was first bracketed against the one unknown, the peer's
error-variance ratio `c`, from measured D/v_o = 0.488 and coverage 0.563:

| c | peer alone ~ | rho | optimal w | projected blind QWK |
|---|---|---|---|---|
| 1.0 | 85.5 | 0.756 | 0.50 | 86.2 |
| 1.1 | 84.0 | 0.769 | 0.40 | 86.0 |
| 1.2 | 82.6 | 0.781 | 0.30 | 85.9 |
| 1.3 | 81.2 | 0.795 | 0.19 | 85.7 |
| >=1.5 | <=78.2 | 0.821 | 0.00 | 85.0-85.4 (a loss) |

That table is arithmetically correct and was still the wrong thing to act on, because it treats
`c` as an uninformed unknown when the pool can be asked directly.

### The measurement that settles it (`s11_calibrate.py`, `s12_calibrate_cv.py`)

Take each pool member in turn, remove it from the ensemble, refit, then blend it back on the
same 56% row mask and measure the realised QWK change against gold. That gives real (x, y) pairs
where `x = D/v_o` is observable for the peer too. Under document-grouped 5-fold CV with the
combiner, scale map and thresholds all refit per fold:

| member | D/v_o | vj/v_o | w=0.10 | w=0.20 | w=0.30 |
|---|---|---|---|---|---|
| `arabertv2_soft_slv` | 0.098 | 1.105 | +0.013 | +0.023 | +0.042 |
| `arabertv2_wkl_slvlr1` | 0.111 | 1.107 | -0.003 | +0.025 | -0.096 |
| `arabertv2_large_emd_slv` | 0.129 | 1.119 | -0.044 | -0.056 | -0.077 |
| `arabertv2_corn_s2` | 0.158 | 1.178 | -0.019 | +0.018 | -0.067 |
| `araelectra_soft_d3` | 0.187 | 1.211 | +0.013 | +0.043 | -0.076 |
| `arbertv2_soft_d3` | 0.222 | 1.214 | +0.110 | +0.108 | +0.082 |
| `camelbert_wkl_d3` | 0.243 | 1.283 | -0.052 | -0.166 | -0.290 |
| `aramodern_soft_raw` | 0.280 | 1.296 | +0.017 | -0.000 | -0.059 |
| `marbert_corn_d3` | 0.319 | 1.308 | +0.019 | -0.078 | -0.135 |
| `aramodern_reg_d3` | 0.441 | 1.481 | -0.020 | -0.228 | -0.405 |
| `arabertv2_large_reg_raw` | 0.612 | 1.662 | -0.048 | -0.168 | -0.415 |

The peer sits at D/v_o = 0.540, inside this range. Its four nearest analogues give 0 of 4
positive at w=0.20 (mean -0.119) and 0 of 4 at w=0.30 (mean -0.253). At w=0.10 it is a wash
(mean -0.008). Aggregates: mean -0.044 at w=0.20, and the correlation between D/v_o and the
realised gain is -0.692, which is the wrong sign for this idea.

Verdict: drop channel B.

### Why the theory misled, and it is not a subtle reason

The variance argument assumes decorrelation is free, that a predictor which disagrees more adds
more independent information. In this pool the opposite holds: members disagree with the
ensemble mainly because they are worse. `vj/v_o` rises monotonically with `D/v_o` (1.11 at 0.098
up to 1.66 at 0.612), and it is `vj/v_o` that drives the sign. Everything at vj/v_o <= 1.21 is
roughly neutral to positive, everything at >= 1.28 loses. So `c` and `D` are not independent
unknowns to sweep over, `D` predicts `c`, and sweeping `c` at fixed `D` explored combinations
that do not occur.

What cannot be resolved: the peer might be distant because it is different rather than worse, a
different pipeline instead of a weaker one. Deciding that needs blind gold. But every analogue
that can be measured says distance means weakness, so the decision that follows the evidence is
not to blend.

### One instrument correction along the way

The first version of this measurement (`s11_calibrate.py`) used a baseline fitted and evaluated
on the same test rows, so it was in-sample optimal and therefore biased against any fixed-weight
perturbation. It reported the blend helping only 1 of 48 members. That bias is real but small,
measured at +0.047 mean at w=0.20, and it does not change the sign. Re-running under proper
grouped CV was still the right call: the dry run had shown the bias flipping a single case from
-0.016 to +0.065, which was enough reason not to trust the cheap instrument.

### Machinery validation (step 5)

The blend mechanism itself was validated against gold before being trusted on blind, by
holding the strongest pool member out of the ensemble entirely and handing it back on a random
53% of test rows as a synthetic peer:

- The variance model recovers rho from D and c as 0.9481 against a true 0.9460, so model error
  is +0.002 and the projection table above is sound given c.
- Realised gain tracked theory (ratio 1.30, both small).
- Under document-grouped 5-fold CV with combiner, scale map and thresholds all refit per fold:
  86.558 -> 86.623 at w = 0.20. A gain even from a near-redundant peer (D/v_o 0.11).
- A single global threshold set survives the induced heteroscedasticity, provided blended rows
  get a small outward push: expansion 1.02 was best or joint-best at 7 of 8 weights.

## Attempt to estimate `c` without labels, and why it was discarded (`s10_estimate_c.py`)

`c` looked identifiable without gold. For predictors of one target, `D_jk = E[(pred_j -
pred_k)^2]` needs no labels, and under a one-factor error model `e_j = f_j g + u_j`,

```
D_jk = psi_j + psi_k + (f_j - f_k)^2
```

48 members give 1,128 equations for ~96 parameters. `D` sees `f` only through differences, so
the common level of `f` is invisible: an error shared identically by every predictor cancels out
of every pairwise distance. That is why one anchor is needed, and it is taken from the
ensemble's own measured 85.50.

It failed its validation. Run on the public test split and checked against the true `v_j`, the
median relative error is 24.5% on the weighted members, and it is biased with structure:

| member | v_true | v_hat | rel |
|---|---|---|---|
| `arabertv2_emd_d3` | 2.808 | 2.013 | -28.3% |
| `arabertv2_wkl_slvlr1` | 2.826 | 1.986 | -29.7% |
| `aramodern_reg_raw` | 3.512 | 4.961 | +41.3% |
| `marbert_corn_d3` | 3.341 | 4.319 | +29.3% |

Predictors near the consensus get their variance under-estimated and predictors far from it get
it over-estimated, because a one-factor model cannot tell different-and-also-good from
different-and-noisy. The peer is the most distant column in the matrix, mean D(member, peer)
2.14 against mean D(member, member) 1.09, so its raw output (`c = 1.98`, peer ~ 71 QWK,
`w* = -0.40`) is an inflated upper bound rather than an estimate. Discarded rather than used.

This is worth recording precisely because the number it produced was actionable-looking and
pointed the opposite way from the earlier bracket. The validation step is the only reason it did
not get believed.

## What this is worth

Channel A only. The shipped file is `sub10_retrieval_only`: sub07 plus public-gold retrieval, 89
rows changed (1.10%), projected 85.55, which is sub07's measured 85.50 plus the retrieval
channel's +0.094 scaled by its 3.02% blind coverage. Channel B was measured and dropped.

`sub10_peerblend` (w=0.20, 808 rows changed) is kept only as the record of a tested and rejected
idea. On the evidence it would score ~85.4, below sub07. It should not be uploaded.

None of this can be verified. The phase closed 2026-08-03 12:00 UTC, before this work was done.
Every number attached to `sub10` is a projection from measured components, not a score.

86-87 was not reachable. 85.5 was the highest blind QWK anyone achieved, and the 86.3 sometimes
cited was the development phase on the easier public test split. The one route that could have
jumped that ceiling, `sents_RL` as an answer key, is definitively closed, and the diverse-peer
route that could have added ~0.5 is closed too, by measurement rather than by assumption.

### Summary of the whole experiment

Three of the four ideas tested here were negative, and two of them looked strongly positive
before being measured properly:

| idea | first impression | after measurement |
|---|---|---|
| `sents_RL` is an answer key | QWK 97.96 vs gold on dev+test | gold copied through where BAREC has labels, blind is model output. No leak. |
| add it as a pool member | in-sample 98.4, grouped CV 98.3 | 92% weight on a column that is gold only on the calibration split. Never. |
| blend it as a decorrelated peer | rho 0.76 vs 0.95, theory says +0.5 to +0.9 | 0 of 4 analogues positive at w=0.20. Drop. |
| retrieve verbatim public gold | small | 95.3% precision, +0.094 on test, +0.05 on blind. Keep. |

The one that survived is the least clever.

## Implementation choices worth recording

- The base is reconstructed from sub07's own stored parameters, and the script refuses to
  proceed unless it reproduces sub07's recorded prediction on all 8,077 rows. It initially matched
  8,075: `meta.json` rounds weights to 5 dp while a greedy multiset assigns exact `k/19`
  rationals, and the rounding walked two rows across a cut. Snapping back to `k/19` gives an
  exact match. Worth knowing generally, since storing rounded weights silently perturbs
  predictions.
- The score-to-label map used to inject a retrieved label is fitted on blind against the base
  system's own predictions, rather than on test against gold. A test-derived slope would import
  the known test->blind shift into the one channel that is otherwise clean. Both are defensible.
  This one avoids a cross-split assumption.
- The peer is de-meaned and sd-matched to the ensemble score on the covered rows before
  blending. A systematic offset between two predictors is bias, not information, and the peer's
  raw mean sits 0.31 levels lower.
- Level 19 is never predicted, by sub10 or by sub07. It carries p ~ 0.0018 in BAREC (~15 of
  8,077 rows expected) and the threshold rule correctly declines to spend a cut on it.
- 808 rows differ from sub07 (10.0%), of which 782 are +-1 and 719 fall on peer-covered rows.
  The marginal is preserved (mean 10.446 vs 10.436, sd 3.053 vs 3.060).

## For the paper

The `sents_RL` result is the publishable part, and it is a methodological warning rather than a
trick: a corpus field that agrees with gold at 97.96 QWK was still not a label. It was gold
where gold existed and model output everywhere else, and the flat-across-splits agreement
profile is what distinguishes the two. Anyone using BAREC-10M for distant supervision needs
that check, or they will train on their own evaluation set and read the result as skill.

The negative result about pool correlation is the other half: with residuals correlated at
0.997 the combination-side methods are used up, and rho = 0.76 against an outside system says
the headroom is in predictor diversity rather than in weighting.

## Files in this directory

The scripts run in order, `s01` to `s12`, and each writes the `.json` next to it. The `.npy`
files are the intermediate arrays they pass between each other, kept so the later steps can be
re-run without redoing the corpus scan:

| file | what it holds |
|---|---|
| `sub07_test_score.npy`, `sub07_blind_score.npy` | the base ensemble's continuous score on the two splits |
| `peer_labels_blind.npy` | `sents_RL` for each blind row, or NaN where the corpus has none |
| `peer_blind_nogold.npy` | the same with every gold-matched row removed, which is the one to use |
| `blind_lookup_hit.npy`, `blind_lookup_label.npy` | channel A: which blind rows matched public gold, and the label retrieved |

The peer arrays are numeric values derived from BAREC-10M (Elmadani et al., CC-BY-SA-4.0)
aligned to the blind row order. They carry no text. Attribution is in
[docs/data.md](../../docs/data.md).

## Reproducing this

Run `s01` to `s12` in order; each writes its json to the scratch root for diffing against the committed one. All ten recorded steps reproduce exactly. `s06`/`s07` record no json; `s07` rebuilds `sub10_peerblend` and reports a 100.00% row match instead of writing.

## Files

| File | Role | Writes |
|---|---|---|
| `blind_lookup_hit.npy` | intermediate array shared between the steps | - |
| `blind_lookup_label.npy` | intermediate array shared between the steps | - |
| `peer_blind_nogold.npy` | intermediate array shared between the steps | - |
| `peer_labels_blind.npy` | intermediate array shared between the steps | - |
| `pool.txt` | the exact member tags this experiment ran on | - |
| `s01_measure.json` | recorded output of `s01_measure.py` | - |
| `s01_measure.py` | exp018 step 1: what is the BAREC-10M peer predictor worth on blind? | scratch root |
| `s02_fingerprint.json` | recorded output of `s02_fingerprint.py` | - |
| `s02_fingerprint.py` | exp018 step 2: are the BAREC-10M labels on blind sentences gold, or model output? | scratch root |
| `s03_decisive.json` | recorded output of `s03_decisive.py` | - |
| `s03_decisive.py` | exp018 step 3: the tie-breaker. Gold copied through, or a peer prediction? | scratch root |
| `s04_exact_match.json` | recorded output of `s04_exact_match.py` | - |
| `s04_exact_match.py` | exp018 step 4: do any blind sentences appear word-for-word in the public gold splits? | scratch root |
| `s05_dryrun.json` | recorded output of `s05_dryrun.py` | - |
| `s05_dryrun.py` | exp018 step 5: test the blending machinery against gold before trusting it on blind | scratch root |
| `s06_adaptive.json` | recorded output of `s06_adaptive.py` | - |
| `s06_adaptive.py` | exp018 step 6: flat peer weight, or weight it by the ensemble's own uncertainty? | scratch root |
| `s07_build_final.py` | exp018 step 7: build the peer-blended submission on top of the 85.5 system | scratch root |
| `s08_knn.json` | recorded output of `s08_knn.py` | - |
| `s08_knn.py` | exp018 step 8: does near-duplicate retrieval extend the override from step 4? | scratch root |
| `s09_trap.json` | recorded output of `s09_trap.py` | - |
| `s09_trap.py` | exp018 step 9: why sents_RL must not go into the pool as a member | scratch root |
| `s10_estimate_c.json` | recorded output of `s10_estimate_c.py` | - |
| `s10_estimate_c.py` | exp018 step 10: estimate the peer's error variance on blind without any gold labels | scratch root |
| `s11_calibrate.json` | recorded output of `s11_calibrate.py` | - |
| `s11_calibrate.py` | exp018 step 11: skip the model, measure what blending at a given D/v_o actually buys | scratch root |
| `s12_calibrate_cv.json` | recorded output of `s12_calibrate_cv.py` | - |
| `s12_calibrate_cv.py` | exp018 step 12: redo step 11 with a measurement that matches the blind setup | scratch root |
| `sub07_blind_score.npy` | intermediate array shared between the steps | - |
| `sub07_test_score.npy` | intermediate array shared between the steps | - |
