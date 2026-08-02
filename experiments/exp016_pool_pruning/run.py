"""exp016: if the pool has stopped improving, is pruning better than growing?

All the pool-size numbers point the same way: 10 -> 39 members was +0.19, 39 -> 66 was
+0.007, 70 -> 71 was -0.08. The likely cause is selection variance: ~70 highly correlated
columns fitted on 7,286 rows, and every extra column is another way for greedy to chase noise.

If that is right, fewer and better should beat more. exp004 tried top-k with uniform weights
and it lost badly (86.28 at k=20 against 86.79 for full-pool greedy), but that changed two
things at once, the pruning and the weighting. This isolates the pruning: rank by solo
calibrated QWK, keep the top N, then run the same capped greedy and prior-shrunk cuts over
what is left.

Members are ranked inside each fold using only that fold's training rows. Ranking on the full
set first would leak the pruning decision into the rows it gets scored on, which is the same
in-sample optimism that already cost one submission.
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

from slra_ot import combiners as C
from slra_ot.members import aligned, doc_structure, pinned_members
from slra_ot.metrics import fast_qwk, full_report
from slra_ot.thresholds import FastQWKThresholds, labels_from_thresholds

AD = ("_ad", "_ad2", "_ad3", "_ad4")


def cv_pruned(X, y, docs, flagged, top_n, cap, k, n_folds=5, seed=0):
    fold = C.doc_folds(docs, n_folds, seed)
    pred = np.empty(len(y), int)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if top_n is not None and top_n < X.shape[1]:
            q = [FastQWKThresholds(rounds=5).fit(X[tr, j], y[tr]).best_qwk_
                 for j in range(X.shape[1])]
            cols = np.argsort(q)[::-1][:top_n]
        else:
            cols = np.arange(X.shape[1])
        Xtr, Xte = X[np.ix_(tr, cols)], X[np.ix_(te, cols)]
        comb = (C.CappedGreedy(flagged[cols], cap=cap, rounds=30) if cap is not None
                else C.GreedyMultiset(rounds=30)).fit(Xtr, y[tr])
        th = C.prior_shrunk_thresholds(comb.score(Xtr), y[tr], k=k)
        pred[te] = labels_from_thresholds(comb.score(Xte), th)
    return fast_qwk(y - 1, pred - 1) * 100, full_report(y, pred)


def main():
    members = pinned_members(HERE)
    tags = [m["tag"] for m in members]
    flagged = np.array([t.endswith(AD) for t in tags])
    ids, X, y = aligned(members, "test")
    docs, _ = doc_structure(ids, "test")
    n_slv = sum("slv" in t for t in tags)
    print("pool %d | %d contaminated | %d silver\n" % (len(tags), flagged.sum(), n_slv))

    rows = []
    print("%6s %5s %5s %8s %6s %8s" % ("top_n", "cap", "k", "CV QWK", "acc19", "implied"))
    for top_n in [10, 16, 24, 36, None]:
        for cap, offset in [(0.30, -1.42), (0.0, -1.40)]:
            q, rep = cv_pruned(X, y, docs, flagged, top_n, cap, 0.35)
            rows.append({"top_n": top_n or len(tags), "cap": cap, "prior_k": 0.35,
                         "cv_qwk": q, "acc19": rep["Acc19"], "implied_blind": q + offset})
            print("%6s %5.2f %5.2f %8.3f %6.2f %8.2f" % (
                ("all" if top_n is None else str(top_n)), cap, 0.35, q, rep["Acc19"], q + offset),
                flush=True)

    rows.sort(key=lambda r: -r["implied_blind"])
    b = rows[0]
    print("\nbest: top_n=%s cap=%.2f CV=%.3f -> implied blind %.2f"
          % (b["top_n"], b["cap"], b["cv_qwk"], b["implied_blind"]))
    print("  sub02/sub04 both scored 85.2; thylinao 85.3; need >=85.5 estimated to submit safely")
    json.dump({"n_pool": len(tags), "results": rows},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
