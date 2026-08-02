"""exp006: is the calibration set the cap, rather than the pool?

exp003 measured a 0.75 QWK gap between in-sample (87.369) and CV (86.623), which is the price
of fitting 45 weights and 18 cuts on 7,286 rows. exp004 then found the 67-member pool doing
worse than the 45-member one under plain greedy, which is the same problem getting worse as
the pool grows.

Selection variance goes down as the set you select on gets bigger, and there is a second
held-out set sitting unused: dev (7,310 sentences) is held out for the 39 members trained on
train alone, though not for the _ad members, which trained on it. So there is a trade:

  P39 + dev+test   39 members, 14,596 calibration rows (2x the rows, 0.59x the pool)
  P66 + test       66 members,  7,286 calibration rows (the current system)

The pool is 66 rather than the 67 exp004 started from: the collapsed member had been removed
by then.

To compare them fairly both are scored on held-out folds of test. The only difference is what
each is allowed to fit on: P39 gets all of dev plus four fifths of test, P66 gets only those
four fifths. Same evaluation rows, same folds.

Later note: the "+0.382 for the _ad members" here was inflation, same as exp003. See exp010.
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
from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402


def eval_procedure(name, Xfit_extra, yextra, Xtest, ytest, docs, factory,
                   n_folds=5, seed=0):
    """CV over test documents. Xfit_extra rows, if any, are always in the training part."""
    fold = C.doc_folds(docs, n_folds, seed)
    pred = np.empty(len(ytest), int)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        Xtr = np.vstack([Xfit_extra, Xtest[tr]]) if len(Xfit_extra) else Xtest[tr]
        ytr = np.concatenate([yextra, ytest[tr]]) if len(yextra) else ytest[tr]
        c = factory().fit(Xtr, ytr)
        th = C.fit_thresholds(c.score(Xtr), ytr)
        pred[te] = labels_from_thresholds(c.score(Xtest[te]), th)
    q = fast_qwk(ytest - 1, pred - 1) * 100
    print(f"{name:44s} CV={q:7.3f}", flush=True)
    return q


def main():
    allm = pinned_members(HERE)
    p39 = [m for m in allm if not m["tag"].endswith(("_ad", "_ad2"))]
    print(f"pool: {len(allm)} total | {len(p39)} train-only (dev genuinely held out)\n")

    test_ids, Xt_all, yt = aligned(allm, "test")
    _, Xt_39, yt2 = aligned(p39, "test")
    dev_ids, Xd_39, yd = aligned(p39, "dev")
    assert (yt == yt2).all()
    docs, _ = doc_structure(test_ids, "test")
    empty = np.empty((0, Xt_all.shape[1]))
    empty39 = np.empty((0, Xt_39.shape[1]))
    ey = np.empty(0, int)

    rows = []
    for cname, factory in [("greedy", lambda: C.GreedyMultiset(rounds=25)),
                           ("bagged greedy", lambda: C.BaggedGreedy(n_bags=15, model_frac=0.5, rounds=15))]:
        rows.append({"system": f"P{len(allm)} / test only", "combiner": cname,
                     "cv_qwk": eval_procedure(f"P{len(allm)} test-only        {cname}",
                                              empty, ey, Xt_all, yt, docs, factory)})
        rows.append({"system": f"P{len(p39)} / test only", "combiner": cname,
                     "cv_qwk": eval_procedure(f"P{len(p39)} test-only        {cname}",
                                              empty39, ey, Xt_39, yt, docs, factory)})
        rows.append({"system": f"P{len(p39)} / dev+test", "combiner": cname,
                     "cv_qwk": eval_procedure(f"P{len(p39)} dev+test (2x cal) {cname}",
                                              Xd_39, yd, Xt_39, yt, docs, factory)})

    rows.sort(key=lambda r: -r["cv_qwk"])
    print(f"\nbest: {rows[0]['system']} + {rows[0]['combiner']} = {rows[0]['cv_qwk']:.3f}"
          f"  -> implied blind {rows[0]['cv_qwk']-1.42:.2f}")
    json.dump({"results": rows}, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
