# SLRA-OT: sentence-level Arabic readability assessment (BAREC 2026, Open Track)

Predict how hard an Arabic sentence is to read, on the 19-level BAREC scale. Systems are
ranked by quadratic weighted kappa (QWK) on an 8,077-sentence blind test set that only the
organisers can score ([BAREC Shared Task 2026, Task 1](https://barec.camel-lab.com/sharedtask2026),
CodaBench [16544](https://www.codabench.org/competitions/16544/)). The track rule, verbatim
from the task site:

> Open Track: No restrictions on external resources, allowing the use of any publicly
> available data.

Result: **blind QWK 85.5**†, rank 1 on leaderboard
([docs/results.md](docs/results.md)). The research question that ended up mattering more than
the ranking: what can and cannot be measured about a system when part of its pool has touched
the calibration split ([docs/findings.md](docs/findings.md)).

Start with `make verify`: it re-derives every uploaded file from the committed caches, row by
row, and recomputes the headline chain forward — 51/51 checks, no GPU, no corpus
([docs/reproduce.md](docs/reproduce.md)).

## What moved the number, in order of measured effect

1. **Silver pretraining** on 439K automatically labelled BAREC-10M sentences: +1.00 QWK on the
   member itself, the only lever that improved members rather than re-weighting them
   ([exp014](experiments/exp014_silver_pretraining/)).
2. **Pool breadth**: +0.19 going from 10 to 39 clean members; flat after that
   ([exp002](experiments/exp002_combiner_bakeoff/)).
3. **Prior-shrunk thresholds**: cuts fitted for QWK, then moved toward the label-prior
   quantiles; +0.17 on dev-to-test, and it stops levels going empty
   ([exp009](experiments/exp009_threshold_regularisation/)).
4. **Dropping the contaminated members** rather than capping them: the decision that separated
   85.5 from 85.3, made against the local estimate and for the honest one
   ([exp017](experiments/exp017_clean_sweep/)).

Thirteen further levers were tried and refuted under the same protocol
([docs/negative-results.md](docs/negative-results.md)). The shipped system is small: 15
weighted members, a greedy weighted mean, one regularised cut rule
([docs/findings.md](docs/findings.md) #10).

## The two findings that decided it

1. **Contamination is invisible to the measurement that suffers from it.** Members that
   early-stopped on the calibration split score 1.06 QWK higher there than they deserve
   (n=18, recomputed by `make verify`), and no estimate computed on that split can see it. The
   local-CV-to-blind gap therefore depends on the contaminated weight: -1.06 clean, -1.36 to
   -1.42 at a 30% cap, -1.90 uncapped ([docs/evaluation.md](docs/evaluation.md)).
2. **Accuracy and QWK pull against each other.** Across teams and across our own submissions,
   systems 10-14 points more accurate at exact-level prediction score about 5 QWK lower, which
   is the metric's algebra, not noise. Acc19 is never a selection proxy here
   ([docs/results.md](docs/results.md)).

## Layout

```
src/slra_ot/        library: metrics, thresholds, combiners, member I/O, models, verify
  cli/              pipeline stages, run as `python -m slra_ot.cli.<stage>`
configs/            backbone ids and the member-training campaigns
experiments/        expNNN_<question>/{README.md, run.py, pool.txt, results.json}
analysis/           diagnostics that are not registered experiments
artifacts/          committed: member score caches, training logs; gitignored: weights, scratch
submissions/        submitted/ (the 8 uploads, as uploaded), candidates/ (built, never uploaded)
docs/               task, compliance, data, method, evaluation, experiments, results,
                    findings, negative-results, members, reproduce, reproducibility
data/               a README only; the corpus is not redistributed
tests/              the threshold optimiser against a reference implementation
```

## Quickstart

```bash
make setup && make verify        # 51/51 checks from the committed caches
python experiments/exp017_clean_sweep/run.py   # any experiment re-runs the same way
```

Retraining the pool (GPU, corpus) is [docs/reproduce.md](docs/reproduce.md) Level 2.

## Experiments

18 experiments, one directory per question, including the refuted ones; the registry with
verdicts is [docs/experiments.md](docs/experiments.md). The ones that carried the result:
exp009 (threshold regularisation), exp010 (contamination measured), exp014 (silver
pretraining), exp017 (clean sweep, the shipped configuration), exp018 (the leak investigation
that cleared BAREC-10M's label field).

## Compliance

The Open Track permits any publicly available data, and every resource used is public;
[docs/compliance.md](docs/compliance.md) holds each component of the shipped system against
the rule, names the transductive axes explicitly, and covers submission limits, registration
and redistribution. Nothing in the shipped chain trains on the calibration split; the members
that do carry zero weight in sub07.

## Licence and data

Code under the [MIT License](LICENSE). BAREC and BAREC-10M are CC-BY-SA-4.0 and are not
redistributed here beyond derived score caches and label vectors
([docs/compliance.md](docs/compliance.md#corpus-redistribution),
[docs/data.md](docs/data.md)).

† Not recomputed by `make verify`. Provenance for each such number is in
[docs/reproducibility.md](docs/reproducibility.md).
