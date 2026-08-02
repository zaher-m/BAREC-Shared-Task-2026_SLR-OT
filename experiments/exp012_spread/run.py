"""exp012: five of six metrics ahead of thylinao, and the ranked one behind.

sub04 vs thylinao on blind:

    Acc19  39.4 vs 37.7      +-1  72.6 vs 70.8
    Acc7   60.8 vs 60.5      Acc5 68.6 vs 66.9     Acc3 75.9 vs 74.6
    QWK    85.2 vs 85.3   <-- the only loss

1.7 points of Acc19 on 8,077 rows is well outside standard error, so this is not noise.
QWK is

    1 - sum_ij W_ij O_ij / sum_ij W_ij E_ij,     E_ij = n_i^true * n_j^pred / N

With the true distribution fixed, the denominator depends only on the predicted distribution,
and it grows as that spreads away from the centre. So two systems making the same errors can
score differently if one predicts the extremes more often. Being more accurate lowers the
numerator, but predicting timidly lowers the denominator faster.

The predicted distribution is narrow: 46% of blind lands in the top three levels and level 19
never gets predicted at all. The threshold optimiser maximises QWK on the calibration set, so
it picks the right spread for test, but the score is compressed on blind (sd 2.735 vs 2.883),
so the same cuts give a narrower distribution there than where they were fitted.

So: sweep a spread factor applied to the target score before thresholding, and see whether QWK
peaks above gamma = 1 on a real cross-sample transfer.

exp007 already found full distribution alignment harmful, but it only tested the single matched
gamma and the two directions behaved very differently (widening -0.04, shrinking -0.24). That
asymmetry is why the whole curve is worth looking at.
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
from slra_ot.members import aligned, pinned_members  # noqa: E402
from slra_ot.metrics import full_report  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

AD = ("_ad", "_ad2", "_ad3", "_ad4")
GAMMAS = [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30]


def spread(s, gamma):
    """Widen (gamma > 1) or squash the score around its own mean."""
    s = np.asarray(s, float)
    return (s - s.mean()) * gamma + s.mean()


def run(name, Xsrc, ysrc, Xtgt, ytgt, rows, thr_k):
    g = C.GreedyMultiset(rounds=25).fit(Xsrc, ysrc)
    s_src, s_tgt = g.score(Xsrc), g.score(Xtgt)
    th = (C.prior_shrunk_thresholds(s_src, ysrc, thr_k) if thr_k is not None
          else C.fit_thresholds(s_src, ysrc))
    ratio = s_tgt.std() / s_src.std()
    print(f"\n{name}  (target/source sd = {ratio:.3f}, thresholds="
          f"{'prior-%.2f' % thr_k if thr_k is not None else 'free'})")
    print(f"{'gamma':>6s} {'QWK':>8s} {'d':>7s} {'acc19':>6s} {'+-1':>6s} {'pred sd':>8s} {'nlev':>5s}")
    base = None
    for gm in GAMMAS:
        p = labels_from_thresholds(spread(s_tgt, gm), th)
        rep = full_report(ytgt, p)
        if base is None or gm == 1.00:
            base = base if gm != 1.00 else rep["QWK"]
        rows.append({"setting": name, "thr_k": thr_k, "gamma": gm, "qwk": rep["QWK"],
                     "acc19": rep["Acc19"], "adj": rep["Adj+-1"], "pred_sd": float(p.std()),
                     "n_levels": int(len(np.unique(p)))})
        print(f"{gm:6.2f} {rep['QWK']:8.3f} {'':>7} {rep['Acc19']:6.2f} {rep['Adj+-1']:6.2f} "
              f"{p.std():8.3f} {len(np.unique(p)):5d}", flush=True)
    for r in rows:
        if r["setting"] == name and r["thr_k"] == thr_k:
            r["delta"] = r["qwk"] - base


def main():
    p39 = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    print(f"{len(p39)} clean members (dev and test both usable)")
    _, Xd, yd = aligned(p39, "dev")
    _, Xt, yt = aligned(p39, "test")

    rows = []
    for thr_k in [None, 0.5]:
        run("dev -> test", Xd, yd, Xt, yt, rows, thr_k)
        run("test -> dev", Xt, yt, Xd, yd, rows, thr_k)

    print("\nmean delta vs gamma=1.00, averaged over both directions:")
    best = None
    for thr_k in [None, 0.5]:
        lbl = "free" if thr_k is None else f"prior-{thr_k}"
        for gm in GAMMAS:
            d = [r["delta"] for r in rows if r["gamma"] == gm and r["thr_k"] == thr_k]
            m = float(np.mean(d))
            print(f"  {lbl:10s} gamma={gm:.2f}  {m:+.3f}")
            if best is None or m > best[0]:
                best = (m, thr_k, gm)
    print(f"\nbest: {'free' if best[1] is None else 'prior-%.1f' % best[1]} gamma={best[2]:.2f} "
          f"mean delta {best[0]:+.3f}")
    print("  -> only worth applying if the gain is consistent in BOTH directions")
    json.dump({"results": rows}, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
