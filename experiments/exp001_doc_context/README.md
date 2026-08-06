# E1. Doc context

**When** Jul 27 · **Verdict** dead: +0.006, the encoders already use the document

## Question

BAREC sentences come from documents, and the blind IDs give the grouping away even though no
metadata ships with them. Does a document-level correction on top of the sentence ensemble buy
anything?

## Method

The frozen 10-member SLRA-ST v5 ensemble and its cached dev/test scores, so this tests the
post-processing alone and needs no GPU. Three corrections on the continuous score (shrink to
document mean, empirical-Bayes shrink, neighbour smoothing), with beta and the window chosen by
document-grouped CV on dev, reported once on test. The diagnostic that decides it comes first:
the intra-document correlation of the residual.

## Recorded outcome

Label ICC within a document is 0.42; residual ICC is 0.07. Best correction on test: +0.006 QWK.
The encoders have already absorbed almost everything the document carries, so there is nothing
left for smoothing to add.

## Reproducing this exactly

`python experiments/exp001_doc_context/run.py`. It reads its pinned pool from `pool.txt` and the cached member scores, and writes `results.json` under `artifacts/regenerated/exp001_doc_context/` for diffing against the committed one. 

## Files

| File | Role | Writes |
|---|---|---|
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp001: does document context add anything on top of the sentence ensemble? | scratch root |
