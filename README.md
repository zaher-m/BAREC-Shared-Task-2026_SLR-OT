# SLRA-OT: sentence-level Arabic readability assessment (BAREC 2026, Open Track)

Predict how hard an Arabic sentence is to read, on the 19-level BAREC scale. Systems are ranked by
quadratic weighted kappa (QWK) on an 8,077-sentence blind test set that only the organisers can
score ([BAREC Shared Task 2026, Task 1](https://barec.camel-lab.com/sharedtask2026)). The track
rule, verbatim from the task site:

> Open Track: No restrictions on external resources, allowing the use of any publicly
> available data.

## Official result

Our system scores **QWK 85.2 / Acc 39.6** on the blind set, **2nd of 4 teams** on the sentence-level
open track, and clears the official baseline by 2.7 QWK.

| | Team | QWK | Acc<sub>19</sub> | ±1 | Dist | Acc<sub>7</sub> | Acc<sub>5</sub> | Acc<sub>3</sub> |
|---|---|---|---|---|---|---|---|---|
| 1 | Thylinao | 85.3 | 37.7 | 70.8 | 1.1 | 60.5 | 66.9 | 74.6 |
| **2** | **ZHR (this repo)** | **85.2** | **39.6** | **72.1** | **1.1** | **61.4** | **68.5** | **75.8** |
| – | *Official baseline* | *82.5* | *42.4* | *73.6* | *1.1* | *61.6* | *68.8* | *75.6* |
| 3 | LSI | 81.7 | 50.6 | 72.9 | 1.0 | 65.0 | 71.2 | 77.2 |
| 4 | Soloeval | 80.6 | 49.1 | 73.6 | 1.1 | 64.5 | 70.0 | 75.1 |

Scores as published on the [shared task results page](https://barec.camel-lab.com/sharedtask2026).
We lead the baseline on the ranked metric and trail it on exact accuracy, which is the metric's
algebra rather than noise: across teams, systems 10 points more accurate at exact-level prediction
score about 5 QWK lower. Acc<sub>19</sub> is never a selection proxy here
Our Strict-track system scores 85.0 and also places 2nd;
it lives in the companion repository
[BAREC-Shared-Task-2026_SLR-ST](https://github.com/zaher-m/BAREC-Shared-Task-2026_SLR-ST).

The system description paper covers both tracks:
*ZHR at BAREC Shared Task 2026: Ordinal-Objective Ensembles with Threshold Calibration and
Silver-Label Pretraining for Arabic Readability Assessment.*

## Data use

The official system trains on the BAREC training split plus the public BAREC-10M silver corpus, from
which every development and test sentence was removed before use. The development split is used for
early stopping, ensemble selection and threshold calibration, which the organizers confirmed is
permitted. The test split is never read. The blind set is used only to produce predictions.

During the evaluation phase we uploaded a system whose selection and cut points were fitted on the
public test split (`sub07`, meta field `fit_on: BAREC public test`). The organizers ruled that
outside the task's data rules. We rebuilt the system on the permitted data, sent a corrected
prediction file on 5 September 2026, and the score above is the organizers' rescoring of that file.
Nothing from the withdrawn system is reported as a result, here or in the paper.

Parts of the negative-results catalog were also measured on the public test split, which was held
out from training for every model family involved. None of those changes was adopted, and the
official system is selected and calibrated on the development split alone.
The repository's internal compliance notes hold the component-by-component audit.

## The official system

[`submissions/candidates/sub11_clean_dev`](submissions/candidates/sub11_clean_dev): greedy selection
on the development split over the 48-member clean pool, keeping 18 weighted members, discretized by
18 cut points fitted for QWK and then shrunk a fraction k = 0.35 toward the label-prior quantiles of
the score distribution. Three silver-pretrained models hold 38% of the total weight.

What moved the number, in order of measured effect:

1. Silver pretraining on 439K automatically labelled BAREC-10M sentences: +1.00 QWK on the member
   itself against a seed-matched twin, the only lever that improved members rather than re-weighting
   them ([exp014](experiments/exp014_silver_pretraining/)).
2. Pool breadth: +0.19 going from 10 to 39 clean members, flat after that
   ([exp002](experiments/exp002_combiner_bakeoff/)).
3. Prior-shrunk thresholds: cuts fitted for QWK, then moved toward the label-prior quantiles. Worth
   +0.17 dev-to-test, and it stops sparse levels going empty
   ([exp009](experiments/exp009_threshold_regularisation/)).
4. Dropping the contaminated members rather than capping them
   ([exp017](experiments/exp017_clean_sweep/)).

The paper reports ten changes that did not help on this track, each measured under a held-out
protocol.

## The finding that outlived the ranking

Contamination is invisible to the measurement that suffers from it. Members that early-stopped on
the calibration split score 1.06 QWK higher there than they deserve (n = 18, recomputed by
`make verify`), and no estimate computed on that split can see it. The local-to-blind gap therefore
depends on the contaminated weight: −1.06 clean, −1.36 to −1.42 at a 30% cap, −1.90 uncapped.
The official pool admits members by what their early
stopping saw, which is why it is called the clean pool.

Start with `make verify`: it re-derives every uploaded file from the committed caches, row by row,
and recomputes the headline chain forward, with no GPU and no corpus.

## Layout

```
src/slra_ot/        library: metrics, thresholds, combiners, member I/O, models, verify
  cli/              pipeline stages, run as `python -m slra_ot.cli.<stage>`
configs/            backbone ids and the member-training campaigns
experiments/        expNNN_<question>/{README.md, run.py, pool.txt, results.json}
analysis/           diagnostics that are not registered experiments
artifacts/          committed: member score caches, training logs; gitignored: weights, scratch
submissions/        submitted/ (the evaluation-phase uploads), candidates/ (built later)
data/               a README only; the corpus is not redistributed
tests/              the threshold optimiser against a reference implementation
```

## Quickstart

```bash
make setup && make verify        # every number re-derived from the committed caches
python experiments/exp017_clean_sweep/run.py   # any experiment re-runs the same way
```

Retraining the pool needs a GPU and the corpus; `make setup` documents the stages.

## Experiments

18 experiments, one directory per question, including the refuted ones, each with its own
README. The ones that carried the result: exp009 (threshold
regularisation), exp010 (contamination measured), exp014 (silver pretraining), exp017 (the clean
sweep), exp018 (the leak investigation that cleared BAREC-10M's label field). Experiments whose
estimates were computed on the public test split are marked as such in the registry; they informed
no part of the official system.

## Licence and data

Code under the [MIT License](LICENSE). BAREC and BAREC-10M are CC-BY-SA-4.0 and are not
redistributed here beyond derived score caches and label vectors.
