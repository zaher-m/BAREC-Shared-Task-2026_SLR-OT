# E11. Ad cap

**When** Aug 2 · **Verdict** cap the _ad members at 30%, taken from blind evidence; shipped as sub04 (85.2); superseded by exp017

## Question

exp010 showed the `_ad` members cannot be trusted to select their own weights, but they are
trained on 13% more data. Instead of dropping them, cap the total weight greedy may give them.

## Method

`combiners.CappedGreedy`: ordinary greedy selection with a hard ceiling on the flagged members'
total weight. The cap is set from the only uncontaminated evidence available, the blind scores
of sub01/sub02/sub03 (0% -> 85.0, ~30% -> 85.2, majority -> 84.9), not from anything computed
on the contaminated split, where no estimator can see the inflation.

## Recorded outcome

The CV column rises with the cap by construction, because it is computed on the split the
flagged members early-stopped on; that rise is the artefact itself, not evidence for a higher
cap. Read Acc19 and the minimum cut gap instead. Shipped as sub04 (cap 30%, prior-0.50 cuts),
blind 85.2, tying sub02.

Later note: dropping the flagged members entirely beat every cap
([exp017](../exp017_clean_sweep/)), so the cap recommendation was wrong and is recorded as
such.

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp011: how much weight should the contaminated members get? | scratch root |
