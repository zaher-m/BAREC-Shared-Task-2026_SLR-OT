"""exp002: which way of combining the 39 members generalises best?

The SLRA-ST system used one recipe (greedy multiset with replacement, cuts by coordinate
ascent). Before spending GPU hours on more members, check whether the combiner itself is
leaving anything on the table.

Only the 39 members trained on train alone take part, so dev is properly held out for all of
them. Every combiner is fitted on dev and scored once on test, with a document-level CV
number on dev next to it to show which methods are just memorising the calibration set.
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
from slra_ot.members import aligned, doc_structure, pinned_members, word_counts  # noqa: E402
from slra_ot.metrics import full_report  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

# This ran before FastQWKThresholds existed (2026-08-02), so both the free cuts and the
# greedy search used the older value grid. Asking for it explicitly is what keeps the numbers
# in results.json reproducible now that the exact optimiser is the default. It costs about 12
# minutes here, which is why the fast one was written.
GRID = 0.02


def main():
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith("_ad")]
    tags = [m["tag"] for m in members]
    dev_ids, Xd, yd = aligned(members, "dev")
    test_ids, Xt, yt = aligned(members, "test")
    dev_docs, _ = doc_structure(dev_ids, "dev")
    print(f"{len(tags)} train-only members | dev {Xd.shape} test {Xt.shape}\n")

    rows = []

    def run(name, combiner, Xd_=Xd, Xt_=Xt, thr="coord"):
        c = combiner.fit(Xd_, yd)
        sd, st = c.score(Xd_), c.score(Xt_)
        th = (C.prior_thresholds(sd, yd) if thr == "prior"
              else C.fit_thresholds(sd, yd, rounds=8, grid=GRID))
        rep = full_report(yt, labels_from_thresholds(st, th))
        dcv = C.cv_qwk(sd, yd, dev_docs) * 100
        rows.append({"method": name, "dev_cv_qwk": dcv, **rep})
        print(f"{name:34s} devCV={dcv:6.3f}  TEST={rep['QWK']:6.3f} "
              f"acc19={rep['Acc19']:5.2f} adj={rep['Adj+-1']:5.2f} mae={rep['MAE']:.3f}", flush=True)
        return c

    # --- baselines to beat --------------------------------------------------
    best_j = int(np.argmax([C.cv_qwk(Xd[:, j], yd, dev_docs) for j in range(len(tags))]))
    print(f"best single by dev-CV: {tags[best_j]}")
    run("single best", C.UniformAvg(cols=[best_j]))
    run("uniform avg (all 39)", C.UniformAvg())

    # --- selection-based ----------------------------------------------------
    g = run("greedy multiset (v5 recipe)", C.GreedyMultiset(rounds=30, grid_objective=True))
    print("   picked:", {tags[j]: round(w * 100) for j, w in enumerate(g.weights_) if w > 0})
    g2 = run("greedy multiset + docCV obj", C.GreedyMultiset(rounds=20, groups=dev_docs, cv_objective=True))
    print("   picked:", {tags[j]: round(w * 100) for j, w in enumerate(g2.weights_) if w > 0})

    # --- regression stackers ------------------------------------------------
    run("nnls", C.NNLS())
    for a in [1.0, 10.0, 100.0]:
        run(f"ridge alpha={a:g}", C.RidgeStack(alpha=a))
    run("hgb", C.HGBStack())

    # stackers plus word count, the one extra feature available on every split
    wd, wt = word_counts(dev_ids, "dev"), word_counts(test_ids, "test")
    Xd2 = np.c_[Xd, np.log1p(wd)]
    Xt2 = np.c_[Xt, np.log1p(wt)]
    run("nnls + log(wc)", C.NNLS(), Xd2, Xt2)
    run("hgb + log(wc)", C.HGBStack(), Xd2, Xt2)

    # --- does the threshold rule change the ranking? ------------------------
    run("greedy multiset / prior thr", C.GreedyMultiset(rounds=30, grid_objective=True),
        thr="prior")

    rows.sort(key=lambda r: -r["dev_cv_qwk"])
    print("\nranked by dev-CV (the only selection signal here that is not in-sample):")
    for r in rows:
        print(f"  {r['method']:34s} devCV={r['dev_cv_qwk']:6.3f}  TEST={r['QWK']:6.3f}")
    json.dump({"n_members": len(tags), "members": tags, "results": rows},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
