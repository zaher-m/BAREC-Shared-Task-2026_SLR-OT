# E2. Combiner bake-off over the 39 train-only members

**When** Jul 27 · **Verdict** combiner saturated; pool breadth is what pays (+0.19)

Question. SLRA-ST shipped one recipe (greedy multiset with replacement + coordinate-ascent
thresholds) chosen without comparison. Is the combiner leaving points on the table, before any
GPU hours go on new members?

Setup. 39 members trained on the BAREC train split only, so DEV is genuinely held out for all
of them. Each combiner is fit on DEV and scored once on TEST.

## Results (TEST QWK is the honest column)

| method | TEST QWK | Acc19 | +-1 | MAE |
|---|---|---|---|---|
| greedy multiset (v5 recipe), full 39 pool | 86.590 | 37.98 | 73.14 | 1.108 |
| greedy multiset + doc-CV objective | 86.553 | 36.80 | 72.50 | 1.123 |
| NNLS | 86.543 | 36.78 | 73.68 | 1.120 |
| greedy multiset / prior-matched thresholds | 86.344 | 41.64 | 71.10 | 1.088 |
| ridge (alpha=1) | 86.250 | 36.58 | 72.54 | 1.138 |
| HGB stack (+log word count) | 86.268 | 35.70 | 74.09 | 1.120 |
| uniform average of all 39 | 86.114 | 37.28 | 70.66 | 1.138 |
| single best (`arabertv2_emd_d3`) | 85.550 | 38.36 | 73.57 | 1.124 |

Reference: the shipped 10-member v5 ensemble scores 86.398 on the same TEST set.

## Read

1. The combiner is not the bottleneck. Greedy multiset already wins; no stacker beats it.
   Widening the pool from the 10 hand-picked v5 members to all 39 is worth +0.19 QWK, more
   than any change of method. Effort belongs in new members, not new combiners.
2. Non-linear stacking actively hurts. HGB and ridge lose 0.3-0.35 to a plain weighted mean.
   With 39 highly-correlated columns there is no non-linear structure left to find, only noise.
3. Prior-matched thresholds trade the ranked metric for one nobody is ranked on. Acc19 jumps
   37.98 -> 41.64 and MAE improves, but QWK drops 0.25. Coordinate ascent stays.
4. `log(word_count)` adds exactly nothing (NNLS identical to 3 decimals). The encoders have
   already priced sentence length in.

## Caveat on the `dev_cv_qwk` column in `results.json`

That column refits only the thresholds per fold, not the combiner. It is therefore valid for
the parameter-free rows (single best, uniform average) and optimistic for every fitted row:
`hgb devCV=91.1` is in-sample leakage, not a real generalisation estimate. It stays in the file
because it is what the run produced, but it must not be used for model selection.
`combiners.cv_qwk_full`, which refits the combiner per fold, is the right tool and is what
exp003 uses.

## Reproducing

`run.py` reads its 39 members from `pool.txt` and asks for the 0.02-grid cuts explicitly
(`GRID`), and its greedy rows ask for the coarse grid objective. All three matter, because the
pool was 45 members at the time and later grew to 76, and both `fit_thresholds` and the greedy
objective moved to the exact optimiser on 2026-08-02 (see exp004). With those pinned the run
reproduces `results.json` exactly. It takes about 12 minutes, nearly all of it in the grid
optimiser, which is what FastQWKThresholds was written to replace.

## Reproducing this exactly

`python experiments/exp002_combiner_bakeoff/run.py`. Reproduces `results.json` exactly except the two `hgb` rows, which move ~0.02 between runs because the gradient-boosted fit is not deterministic across threads.

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp002: which way of combining the 39 members generalises best? | scratch root |
