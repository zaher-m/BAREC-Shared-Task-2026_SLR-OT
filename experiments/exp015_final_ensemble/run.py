"""exp015: final selection, now with the silver members in the pool.

Runs over whatever members exist when it is called, so it can be re-run as the silver queue
finishes more of them.

Everything else is fixed by the earlier experiments:

  pool       the _ad members are contaminated on the calibration split (1.06 QWK of
             inflation, exp010), so cap them at 30%, which is where the three blind results
             put the optimum (0% -> 85.0, 30% -> 85.2, unrestricted -> 84.9). Cap 0 and
             uncapped are also run, as a sanity band.
  combiner   greedy multiset. Stacking loses (exp002), bagging adds +0.004 once the collapsed
             member is gone (exp004).
  thresholds prior-shrunk. Free cuts overfit and emptied a level on blind (exp009).
  estimate   document-grouped 5-fold CV on test, combiner and cuts refitted per fold, minus
             the offset that matches the pool's contamination level.

The silver members use the clean protocol, so adding them does not change the offset.
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
from slra_ot.thresholds import FastQWKThresholds, labels_from_thresholds  # noqa: E402

AD = ("_ad", "_ad2", "_ad3", "_ad4")


def cv(factory, X, y, docs, k, n_folds=5, seed=0):
    fold = C.doc_folds(docs, n_folds, seed)
    pred = np.empty(len(y), int)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        c = factory().fit(X[tr], y[tr])
        th = C.prior_shrunk_thresholds(c.score(X[tr]), y[tr], k=k)
        pred[te] = labels_from_thresholds(c.score(X[te]), th)
    return fast_qwk(y - 1, pred - 1) * 100, full_report(y, pred)


def main():
    members = pinned_members(HERE)
    tags = [m["tag"] for m in members]
    flagged = np.array([t.endswith(AD) for t in tags])
    silver = [t for t in tags if t.endswith("_slv") or "_slv" in t]
    ids, X, y = aligned(members, "test")
    docs, _ = doc_structure(ids, "test")
    print(f"pool {len(tags)} | {flagged.sum()} contaminated | {len(silver)} silver: {silver}\n")

    print("solo calibrated TEST QWK, best 8 members:")
    solo = sorted(((FastQWKThresholds(rounds=6).fit(X[:, j], y).best_qwk_ * 100, tags[j])
                   for j in range(len(tags))), reverse=True)
    for q, t in solo[:8]:
        mark = "  <-- SILVER" if t in silver else ("  (contaminated)" if t.endswith(AD) else "")
        print(f"  {q:6.2f}  {t}{mark}")

    rows = []
    print(f"\n{'pool':>10s} {'k':>5s} {'CV QWK':>8s} {'acc19':>6s} {'+-1':>6s} {'implied blind':>14s}")
    for cap, label, offset in [(0.0, "clean", -1.40), (0.30, "cap30", -1.42), (None, "free", -1.90)]:
        for k in [0.35, 0.50, 0.65]:
            if cap is None:
                fac = lambda: C.GreedyMultiset(rounds=30)
            else:
                fac = lambda cap=cap: C.CappedGreedy(flagged, cap=cap, rounds=30)
            q, rep = cv(fac, X, y, docs, k)
            rows.append({"pool": label, "cap": cap, "prior_k": k, "cv_qwk": q,
                         "acc19": rep["Acc19"], "adj": rep["Adj+-1"],
                         "implied_blind": q + offset})
            print(f"{label:>10s} {k:5.2f} {q:8.3f} {rep['Acc19']:6.2f} {rep['Adj+-1']:6.2f} "
                  f"{q+offset:14.2f}", flush=True)

    rows.sort(key=lambda r: -r["implied_blind"])
    b = rows[0]
    print(f"\nbest by implied blind: pool={b['pool']} k={b['prior_k']} "
          f"CV={b['cv_qwk']:.3f} -> {b['implied_blind']:.2f}   (sub02/sub04 = 85.2, leader = 85.3)")
    print("\noffsets are per-pool because contamination changes them: "
          "-1.40 clean, -1.42 at 30% cap, -1.90 uncapped (measured on sub01/02/03).")
    json.dump({"n_pool": len(tags), "silver": silver, "results": rows},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
