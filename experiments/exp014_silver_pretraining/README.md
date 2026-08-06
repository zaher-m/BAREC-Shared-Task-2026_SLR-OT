# E14. Silver-label pretraining on BAREC-10M

**When** Aug 2 - Aug 3 · **Verdict** silver pretraining works: +1.00 QWK on the member itself

Why this and nothing else. By this point every post-processing avenue was measured and dead
(exp001 document context, exp002 stacking, exp005 normalisation, exp007 alignment, exp012
spread, exp013 uncertainty shrinkage, BAREC-10M metadata, exp004 bagging). Four different
recipes had all landed at 85.0-85.2 on blind. That pattern says the member pool is the
ceiling, not the way members are combined, so the only thing worth GPU time was a member
built differently.

## The resource

`CAMeL-Lab/BAREC-10M`, public and CC-BY-SA-4.0, so allowed in the Open Track. Ships ~552K
sentences with automatic 19-level readability labels, plus precomputed morphology.
After filtering: 438,996 usable sentences, 8x the 54,845 gold training sentences, with a
label histogram matching BAREC's shape.

Two practical wins:
- It ships precomputed d3tok, saving a ~2h CAMeL Tools disambiguation run. Converting their
  diacritised format to the one the members use (split clitic markers, strip harakat,
  alef-wasla -> alef) matches exactly on 70.5% of overlapping sentences. The rest differ by
  genuine morphological-analysis choices rather than formatting, which is acceptable noise
  for pretraining.
- 59/60 sampled blind sentences are present in the corpus, confirming it covers the blind
  documents' sources.

## The exclusion that makes this safe

BAREC-10M contains BAREC, so it contains the DEV and TEST sentences. Pretraining on them,
even with silver rather than gold labels, would leak into the two sets used for early
stopping and for ensemble selection and threshold calibration. That is sub03's failure
arriving by a different route.

All 22,852 dev/test sentences were dropped from the silver corpus by normalised text match.
Blind sentences were deliberately not dropped: they carry no gold label anywhere, nothing
measured locally is computed on them, and the corpus is public, which makes it ordinary
transduction rather than leakage.

Gold fine-tuning uses the clean protocol (train split, early stopping on DEV), so TEST stays
untouched by training and by model selection, end to end.

## Result: it works

Stage 1: one epoch over 438,996 sentences, 3,875 s. The gold stage then peaks at epoch 0
and declines (84.08 -> 83.56 -> 83.23 -> 82.96), which is what a model that already converged
on the task does at lr 2e-5; early stopping keeps ep0.

| member | dev QWK (naive) peak | solo test QWK |
|---|---|---|
| `arabertv2_emd_c2` (fresh twin) | 82.79 | 84.93 |
| `arabertv2_emd_slv` (silver) | 84.08 | 85.93 |
| `arabertv2_soft_c2` (fresh, other objective) | 83.45 | 85.37 |
| `arabertv2_emd_d3` (previous pool best) | - | 86.07 |

Silver pretraining is worth +1.00 QWK over its direct twin, making it the third-strongest
member in a 70-member pool and the best clean one after `emd_d3`. That is a genuine change in
member quality rather than a re-weighting of information the pool already had, which is what
it needed once every combination-side lever was exhausted.

### Two measurement traps hit while reading this result

1. dev-naive is not a proxy for calibrated test. ep0 dev-naive 84.08 looked like it beat
   every fresh model's peak; the calibrated test gain over the twin is +1.00, real but
   smaller than that framing implied.
2. Solo QWK depends on where the thresholds were fit. The training log prints 85.36
   (thresholds fit on dev, applied to test); the pool audit prints 85.93 (thresholds fit on
   test), which is how every other member's solo number is computed. Only the second is
   comparable to the 84.93 / 86.07 figures above. Always state the threshold split.

## Ensemble effect

Adding this one member to the clean pool moves document-grouped CV from 86.523 -> 86.635,
and the capped-30% pool reaches 86.713 (implied blind 85.29) against 85.2 for the two
submissions that actually scored. One member is worth roughly +0.10 to the ensemble.

## Follow-on

`configs/campaigns/05_silver_pretraining.yaml` queues three more members (soft and wkl on
AraBERTv2, corn on AraELECTRA) plus an emd reseed. It was deliberately not launched until this
first result was in hand: launching a multi-hour queue on an unverified assumption is what
produced sub03.

`slra_ot.cli.train_silver` caches the stage-1 weights per objective and variant
(`--init_silver`), so a member that differs only in its gold hyper-parameters skips the
65-minute pretraining pass.

## Reproducing this exactly

Notes only; the experiment is a training run. `python -m slra_ot.cli.run_campaign configs/campaigns/05_silver_pretraining.yaml` retrains the members (GPU, hours); the comparison numbers recompute from the cached scores and the training logs.

