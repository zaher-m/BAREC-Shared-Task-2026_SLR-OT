# E8. Combiner tuning

**When** Aug 2 · **Verdict** flat: nothing to win from combiner hyper-parameters

## Question

With the pool audited (66 members), do combiner hyper-parameters matter: greedy rounds, bagging
fraction, seed averaging, or blending greedy with NNLS?

## Recorded outcome

```
bagged m.25 s1               86.852
bagged m.35 x3 seeds         86.804
bagged m.50 s1               86.796
blend greedy+bagged+nnls     86.795
greedy r25                   86.792
greedy r40                   86.756
bagged m.35 s1               86.736
```

A 0.12 spread across everything tried, with no structure. The combiner is saturated, which is
exp002's conclusion holding at 66 members.

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp008: tune the combiner on the clean 66-member pool, and check what sub03 shipped | scratch root |
