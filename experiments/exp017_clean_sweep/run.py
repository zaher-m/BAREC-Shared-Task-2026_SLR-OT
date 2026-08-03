"""exp017: the offset depends on the pool, so sweep the clean pool properly.

sub06 and sub07 settled something the calibration set could never answer:

    sub06  cap30 (30% _ad weight)   CV 86.663 -> blind 85.3   offset -1.36
    sub07  clean only               CV 86.558 -> blind 85.5   offset -1.06

We expected sub06 to come out ahead, and it did not. The contaminated members do not inflate the estimate by a
fixed amount, they inflate it by more than they help, so comparing a clean pool against a
capped one by CV is misleading in a consistent direction. The 30% cap was trading a real signal
for an artefact.

So: only rank clean configurations against each other, where the offset is about -1.06.
Improving on 85.5 needs clean CV >= ~86.56.
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
from slra_ot.thresholds import labels_from_thresholds

AD = ("_ad", "_ad2", "_ad3", "_ad4")
CLEAN_OFFSET = -1.06


def cv(fac, X, y, docs, k, n_folds=5, seed=0):
    fold = C.doc_folds(docs, n_folds, seed)
    pred = np.empty(len(y), int)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        c = fac().fit(X[tr], y[tr])
        th = (C.prior_shrunk_thresholds(c.score(X[tr]), y[tr], k=k) if k is not None
              else C.fit_thresholds(c.score(X[tr]), y[tr]))
        pred[te] = labels_from_thresholds(c.score(X[te]), th)
    return fast_qwk(y - 1, pred - 1) * 100, full_report(y, pred)


def main():
    clean = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in clean]
    ids, X, y = aligned(clean, "test")
    docs, _ = doc_structure(ids, "test")
    print("clean pool: %d members (%d silver)\n" % (len(tags), sum("slv" in t for t in tags)))
    print("sub07 reference: CV 86.558 -> blind 85.5 (offset %.2f)" % CLEAN_OFFSET)
    print("to beat 85.5 need CV >= %.3f\n" % (85.5 - CLEAN_OFFSET))

    combiners = [
        ("greedy r30", lambda: C.GreedyMultiset(rounds=30)),
        ("greedy r20", lambda: C.GreedyMultiset(rounds=20)),
        ("bagged m.5", lambda: C.BaggedGreedy(n_bags=20, model_frac=0.5, rounds=15)),
        ("bagged m.35", lambda: C.BaggedGreedy(n_bags=20, model_frac=0.35, rounds=15)),
    ]
    rows = []
    print("%-13s %6s %8s %6s %9s" % ("combiner", "k", "CV QWK", "acc19", "implied"))
    for cname, fac in combiners:
        for k in [None, 0.25, 0.35, 0.50]:
            q, rep = cv(fac, X, y, docs, k)
            rows.append({"combiner": cname, "prior_k": k, "cv_qwk": q,
                         "acc19": rep["Acc19"], "implied": q + CLEAN_OFFSET})
            print("%-13s %6s %8.3f %6.2f %9.2f" % (
                cname, "free" if k is None else "%.2f" % k, q, rep["Acc19"], q + CLEAN_OFFSET),
                flush=True)

    rows.sort(key=lambda r: -r["cv_qwk"])
    b = rows[0]
    print("\nBEST clean: %s k=%s  CV=%.3f -> implied %.2f" % (
        b["combiner"], b["prior_k"], b["cv_qwk"], b["implied"]))
    json.dump({"n_clean": len(tags), "clean_offset": CLEAN_OFFSET, "results": rows},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
