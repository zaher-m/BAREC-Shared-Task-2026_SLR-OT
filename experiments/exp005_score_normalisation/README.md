# E5. Score normalisation

**When** Aug 2 · **Verdict** dead: z-scoring costs 0.10, spreads only vary 1.21x

## Question

Members output continuous levels on slightly different scales (regression scalars vs softmax
expectations). Does putting them on a common scale before averaging help?

## Method

Same document-grouped CV as exp004, on the 66-member post-audit pool: raw scores vs per-column
z-scoring vs rank-normalisation, under greedy and bagged-greedy selection.

## Recorded outcome

```
raw scores       greedy         CV= 86.792
raw scores       bagged greedy  CV= 86.796
z-scored         greedy         CV= 86.692
rank-normalised  bagged greedy  CV= 86.876
```

Member score spreads vary only 1.21x, so there is little to correct, and z-scoring throws away
scale information the cuts were using (-0.10). Rank-normalisation with bagging shows +0.08,
which is inside the noise band exp016 later established for this CV and was not pursued.

## Reproducing this exactly

`python experiments/exp005_score_normalisation/run.py`

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp005: do the members need putting on a common scale before averaging? | scratch root |
