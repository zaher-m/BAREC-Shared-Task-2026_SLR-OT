# E17. Clean sweep

**When** Aug 3 · **Verdict** rank clean against clean; basis for sub07 (85.5), sub08, sub09

## Question

sub06 (capped) lost to sub07 (clean), settling something the calibration set could never
answer: the contaminated members inflate the estimate by more than they help. So what does the
clean pool support, ranked properly against itself?

## Method

Document-grouped 5-fold CV over the 47 clean members, sweeping combiner (greedy r20/r30, bagged
two ways) against threshold shrinkage k in {free, 0.25, 0.35, 0.50}, with the clean offset
-1.06 (measured on sub07) turning each CV into an implied blind score.

## Recorded outcome

```
greedy r30  k=0.25  CV 86.561
greedy r30  k=0.35  CV 86.558   <- sub07's configuration
greedy r30  k=0.50  CV 86.553
greedy r30  free    CV 86.468
```

The top of the sweep is flat to 0.01, so the choice among k in 0.25-0.50 was not resolvable
locally; blind said 85.5 / 85.4 / 85.3 for k = 0.35 / 0.25 / 0.80 (sub07/sub08/sub09).

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp017: the offset depends on the pool, so sweep the clean pool properly | scratch root |
