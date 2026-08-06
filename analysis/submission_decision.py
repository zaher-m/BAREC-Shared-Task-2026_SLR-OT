"""Is the current pool worth spending a submission on, or is it better to wait?

Prints the two numbers that decide it: the new member's solo calibrated test QWK against the
current pool best, and the best CV estimate with the blind score it implies. Then applies the
rule we settled on, which is to only submit at implied >= 85.5. Across four blind results the
estimates were 0 to 0.5 too optimistic and never too pessimistic, and thylinao was at 85.3,
so anything below that bar is a coin flip.

    python analysis/submission_decision.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slra_ot import combiners as C                              # noqa: E402
from slra_ot.members import aligned, doc_structure, load_members  # noqa: E402
from slra_ot.metrics import fast_qwk, full_report               # noqa: E402
from slra_ot.thresholds import FastQWKThresholds, labels_from_thresholds  # noqa: E402

AD = ("_ad", "_ad2", "_ad3", "_ad4")
BAR = 85.5
TOP_SCORE = 85.3          # thylinao's blind QWK at the time


def cv(fac, X, y, docs, k, n_folds=5, seed=0):
    fold = C.doc_folds(docs, n_folds, seed)
    pred = np.empty(len(y), int)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        c = fac().fit(X[tr], y[tr])
        th = C.prior_shrunk_thresholds(c.score(X[tr]), y[tr], k=k)
        pred[te] = labels_from_thresholds(c.score(X[te]), th)
    return fast_qwk(y - 1, pred - 1) * 100, full_report(y, pred)


def main():
    ms = load_members()
    tags = [m["tag"] for m in ms]
    flagged = np.array([t.endswith(AD) for t in tags])
    ids, X, y = aligned(ms, "test")
    docs, _ = doc_structure(ids, "test")
    solo = sorted(((FastQWKThresholds(rounds=6).fit(X[:, j], y).best_qwk_ * 100, tags[j])
                   for j in range(len(tags))), reverse=True)
    print(f"pool {len(tags)} members\n\ntop 6 by solo calibrated TEST QWK:")
    for q, t in solo[:6]:
        print(f"  {q:6.2f}  {t}{'   <-- SILVER' if 'slv' in t else ''}")

    rows = []
    print(f"\n{'pool':>8s} {'k':>5s} {'CV QWK':>8s} {'acc19':>6s} {'implied':>9s}")
    for cap, lbl, off in [(0.30, "cap30", -1.45), (0.0, "clean", -1.40)]:
        for k in [0.35, 0.50]:
            fac = (lambda cap=cap: C.CappedGreedy(flagged, cap=cap, rounds=30))
            q, rep = cv(fac, X, y, docs, k)
            rows.append((q + off, q, lbl, k, rep))
            print(f"{lbl:>8s} {k:5.2f} {q:8.3f} {rep['Acc19']:6.2f} {q+off:9.2f}", flush=True)

    rows.sort(reverse=True)
    imp, q, lbl, k, rep = rows[0]
    print(f"\nBEST: pool={lbl} k={k:.2f}  CV={q:.3f}  implied blind={imp:.2f}")
    print(f"top score {TOP_SCORE} | submit bar {BAR:.1f}")
    print("\nRECOMMENDATION: " + (
        "SUBMIT, clears the bar with room for the usual optimism" if imp >= BAR else
        f"HOLD, implied {imp:.2f} is below the {BAR:.1f} bar. Submitting is a coin flip "
        f"against {TOP_SCORE}"))


if __name__ == "__main__":
    main()
