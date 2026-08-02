"""exp013: shrink the confident-but-wrong tail, per sentence.

Looking at the blind metrics again, sub04 vs thylinao:

    Acc19  39.4 vs 37.7   (ahead)      +-1  72.6 vs 70.8   (ahead)
    MAE     1.1 vs  1.1   (equal)       QWK  85.2 vs 85.3   (behind)

Better on exact hits and better within +-1, but the same mean absolute error. The error has to
be somewhere, so it must be a heavier far tail. QWK squares errors, so a handful of off-by-6
predictions outweighs a lot of extra exact hits. exp012 ruled out a global spread fix
(gamma = 1.00 was optimal), so this is not a scale problem, it is specific sentences the
ensemble gets badly wrong.

A weighted mean goes wrong where its members disagree. Under squared loss the best point
prediction for an uncertain case is pulled toward the mean, by an amount set by how uncertain
it is, and QWK's numerator is a squared loss. So:

    s' = m + (s - m) * tau^2 / (tau^2 + lam * var_i)

with var_i the across-member variance for sentence i and m the pool mean. Confident sentences
are left alone and the ones the pool argues about get pulled in, which should be where the
over-confident extreme predictions live.

Different from exp012 because it is per-sentence: the global gamma had to trade accuracy
everywhere to buy tail safety anywhere. Checked in both transfer directions, and only worth
applying if it helps in both.
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
LAMS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8]


def member_var(X, w):
    """Weighted variance across members, per sentence."""
    w = np.asarray(w, float) / np.sum(w)
    mu = X @ w
    return (X - mu[:, None]) ** 2 @ w


def shrink(s, var, lam, centre, tau2):
    if lam == 0.0:
        return s
    f = tau2 / (tau2 + lam * var)
    return centre + (s - centre) * f


def tail_profile(y, p):
    e = np.abs(np.asarray(p, int) - np.asarray(y, int))
    return {f"err>={k}": float((e >= k).mean() * 100) for k in (2, 4, 6)}


def run(name, Xsrc, ysrc, Xtgt, ytgt, rows):
    g = C.GreedyMultiset(rounds=25).fit(Xsrc, ysrc)
    w = g.weights_
    s_src, s_tgt = g.score(Xsrc), g.score(Xtgt)
    v_src, v_tgt = member_var(Xsrc, w), member_var(Xtgt, w)
    centre = float(s_src.mean())
    tau2 = float(s_src.var())
    print(f"\n{name}: member-variance mean src {v_src.mean():.3f} tgt {v_tgt.mean():.3f}")
    print(f"{'lam':>5s} {'QWK':>8s} {'acc19':>6s} {'+-1':>6s} {'MAE':>6s} "
          f"{'e>=2':>6s} {'e>=4':>6s} {'e>=6':>6s}")
    base = None
    for lam in LAMS:
        # refit the cuts on the shrunk source so source and target match
        ss = shrink(s_src, v_src, lam, centre, tau2)
        st = shrink(s_tgt, v_tgt, lam, centre, tau2)
        th = C.prior_shrunk_thresholds(ss, ysrc, 0.5)
        p = labels_from_thresholds(st, th)
        rep = full_report(ytgt, p)
        tp = tail_profile(ytgt, p)
        if lam == 0.0:
            base = rep["QWK"]
        rows.append({"setting": name, "lam": lam, "qwk": rep["QWK"], "delta": rep["QWK"] - base,
                     "acc19": rep["Acc19"], "adj": rep["Adj+-1"], "mae": rep["MAE"], **tp})
        print(f"{lam:5.2f} {rep['QWK']:8.3f} {rep['Acc19']:6.2f} {rep['Adj+-1']:6.2f} "
              f"{rep['MAE']:6.3f} {tp['err>=2']:6.2f} {tp['err>=4']:6.2f} {tp['err>=6']:6.2f}",
              flush=True)


def main():
    p39 = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    print(f"{len(p39)} clean members")
    _, Xd, yd = aligned(p39, "dev")
    _, Xt, yt = aligned(p39, "test")

    rows = []
    run("dev -> test", Xd, yd, Xt, yt, rows)
    run("test -> dev", Xt, yt, Xd, yd, rows)

    print("\nmean delta over both directions:")
    best = (None, -9)
    for lam in LAMS:
        d = [r["delta"] for r in rows if r["lam"] == lam]
        m = float(np.mean(d))
        ok = all(x > -0.02 for x in d)
        print(f"  lam={lam:4.2f}  {m:+.3f}   per-direction {[f'{x:+.3f}' for x in d]}"
              f"{'' if ok else '   (not consistent)'}")
        if m > best[1] and ok:
            best = (lam, m)
    print(f"\n-> {'apply lam=%.2f (mean %+.3f)' % (best[0], best[1]) if best[0] else 'no consistent gain'}")
    json.dump({"results": rows, "best_lam": best[0]},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
