"""exp011: how much weight should the contaminated members get?

The _ad members are awkward. They are trained on 13% more data, so as models they should be
at least as good as the clean ones. But they early-stopped on the public test split, which is
also the selection and calibration set, so their scores there are about 1.06 QWK too high and
nothing computed on that split can see it. Greedy buys too many of them, and the more it buys
the worse blind gets.

The only evidence that is not contaminated is the leaderboard itself:

    sub01   0% weight on _ad    -> 85.0
    sub02   ~30% weight         -> 85.2   (best so far)
    sub03   majority weight     -> 84.9

which puts the optimum somewhere around 30%. This experiment does not try to re-derive the cap
from the contaminated split, since that cannot work. It checks the weaker question the split
can answer: with a cap in place, does everything else still behave, and how does the estimate
move as the cap changes?

Careful with the CV column. It is computed on test, so it is inflated in proportion to the
cap. CV rising with the cap is expected and is not evidence the cap should be higher. The
columns to read are Acc19 (the failure that sank sub03) and the minimum gap between cuts.

Later note: sub07 showed that dropping the flagged members beats any cap. See exp017.
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


def cv_capped(X, y, docs, flagged, cap, k, n_folds=5, seed=0):
    fold = C.doc_folds(docs, n_folds, seed)
    pred = np.empty(len(y), int)
    ws = []
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        c = (C.CappedGreedy(flagged, cap=cap, rounds=25) if cap is not None
             else C.GreedyMultiset(rounds=25)).fit(X[tr], y[tr])
        ws.append(getattr(c, "flagged_weight_", float(c.weights_[flagged].sum())))
        th = C.prior_shrunk_thresholds(c.score(X[tr]), y[tr], k=k)
        pred[te] = labels_from_thresholds(c.score(X[te]), th)
    return fast_qwk(y - 1, pred - 1) * 100, full_report(y, pred), float(np.mean(ws))


def main():
    members = pinned_members(HERE)
    tags = [m["tag"] for m in members]
    flagged = np.array([t.endswith(AD) for t in tags])
    ids, X, y = aligned(members, "test")
    docs, _ = doc_structure(ids, "test")
    print(f"pool {len(tags)} | {flagged.sum()} contaminated (early-stopped on TEST)\n")
    print(f"{'cap':>6s} {'k':>5s} {'CV QWK*':>8s} {'acc19':>6s} {'+-1':>6s} {'MAE':>6s} {'ad_w':>6s}")

    rows = []
    for cap in [0.0, 0.20, 0.30, 0.40, None]:
        for k in [0.5]:
            q, rep, adw = cv_capped(X, y, docs, flagged, cap, k)
            rows.append({"cap": cap, "prior_k": k, "cv_qwk_inflated": q,
                         "acc19": rep["Acc19"], "adj": rep["Adj+-1"], "mae": rep["MAE"],
                         "mean_ad_weight": adw})
            label = "free" if cap is None else f"{cap:.2f}"
            print(f"{label:>6s} {k:5.2f} {q:8.3f} {rep['Acc19']:6.2f} {rep['Adj+-1']:6.2f} "
                  f"{rep['MAE']:6.3f} {adw*100:5.1f}%", flush=True)

    print("\n* CV is computed on the split the flagged members early-stopped on, so it is")
    print("  inflated in proportion to the cap. Rising CV with rising cap is an artefact.")
    print("  The cap is chosen from the three blind measurements (0% -> 85.0, 30% -> 85.2,")
    print("  free -> 84.9), not from this column.")
    json.dump({"results": rows}, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
