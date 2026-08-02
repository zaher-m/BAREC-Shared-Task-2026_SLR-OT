"""exp010: rebuild on the uncontaminated pool, with the threshold rule that transfers.

Two corrections behind this:

1. The _ad members are contaminated on the calibration set. They early-stopped on the public
   test split, which is the same set used for selection and for fitting the cuts. Measured
   inflation, best epoch minus mean epoch: 1.06 QWK. So any estimate computed on test
   overstates an _ad-heavy ensemble, and it overstates it more the more weight those members
   carry. That matches the offsets: sub01 (no _ad) -1.40, sub02 (4 weighted) -1.42, sub03
   (many) -1.90. exp003's "+0.180 from the _ad members" and exp006's "+0.382" were both
   measuring this inflation rather than skill.

2. Free cuts overfit (exp009). Shrinking halfway to the label prior is worth +0.17 QWK and
   +2.2 Acc19 on dev->test and stops levels going empty.

The clean pool is the 39 members trained on the train split alone. They early-stopped on dev,
so dev is contaminated for them but test is untouched by both training and selection. Fitting
and calibrating on test is therefore fine, and document-grouped CV on test estimates the whole
procedure.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from slra_ot.paths import REGENERATED  # noqa: E402

# re-runs write here, never over the committed record
OUT_DIR = os.path.join(str(REGENERATED), os.path.basename(HERE))
os.makedirs(OUT_DIR, exist_ok=True)

from slra_ot import combiners as C  # noqa: E402
from slra_ot.members import aligned, doc_structure, pinned_members  # noqa: E402
from slra_ot.metrics import fast_qwk, full_report  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

AD = ("_ad", "_ad2", "_ad3", "_ad4")


def cv(factory, X, y, docs, thr_rule, n_folds=5, seed=0):
    """Document-grouped CV on test, refitting both the combiner and the cuts per fold."""
    fold = C.doc_folds(docs, n_folds, seed)
    pred = np.empty(len(y), int)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        c = factory().fit(X[tr], y[tr])
        s_tr = c.score(X[tr])
        th = thr_rule(s_tr, y[tr])
        pred[te] = labels_from_thresholds(c.score(X[te]), th)
    return fast_qwk(y - 1, pred - 1) * 100, full_report(y, pred)


def main():
    allm = pinned_members(HERE)
    clean = [m for m in allm if not m["tag"].endswith(AD)]
    print(f"{len(allm)} members total | {len(clean)} clean (TEST never used in training "
          f"or early stopping)\n")

    ids, X, y = aligned(clean, "test")
    docs, _ = doc_structure(ids, "test")

    rules = {
        "free (sub03 rule)": lambda s, yy: C.fit_thresholds(s, yy),
        "prior-0.25": lambda s, yy: C.prior_shrunk_thresholds(s, yy, 0.25),
        "prior-0.50": lambda s, yy: C.prior_shrunk_thresholds(s, yy, 0.50),
        "prior-0.65": lambda s, yy: C.prior_shrunk_thresholds(s, yy, 0.65),
    }
    combiners = {
        "greedy": lambda: C.GreedyMultiset(rounds=25),
        "bagged m.5": lambda: C.BaggedGreedy(n_bags=20, model_frac=0.5, rounds=15),
    }

    rows = []
    print(f"{'combiner':12s} {'threshold rule':18s} {'CV QWK':>8s} {'acc19':>6s} {'+-1':>6s} {'MAE':>6s}")
    for cn, cf in combiners.items():
        for rn, rf in rules.items():
            q, rep = cv(cf, X, y, docs, rf)
            rows.append({"combiner": cn, "rule": rn, "cv_qwk": q, "acc19": rep["Acc19"],
                         "adj": rep["Adj+-1"], "mae": rep["MAE"]})
            print(f"{cn:12s} {rn:18s} {q:8.3f} {rep['Acc19']:6.2f} {rep['Adj+-1']:6.2f} "
                  f"{rep['MAE']:6.3f}", flush=True)

    rows.sort(key=lambda r: -r["cv_qwk"])
    b = rows[0]
    print(f"\nbest on the clean pool: {b['combiner']} + {b['rule']}  CV={b['cv_qwk']:.3f}")
    print(f"  clean-pool offset is the sub01/sub02 one (-1.40) -> implied blind "
          f"{b['cv_qwk']-1.40:.2f}   (need > 85.30, sub02 = 85.2)")
    json.dump({"n_clean": len(clean), "clean_tags": [m["tag"] for m in clean], "results": rows},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
