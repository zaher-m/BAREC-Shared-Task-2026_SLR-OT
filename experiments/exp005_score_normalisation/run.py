"""exp005: do the members need putting on a common scale before averaging?

Every member outputs a continuous level, but not the same way. reg emits a regression
scalar, corn a sum of cumulative sigmoids, and soft/wkl/emd an expectation over a 19-way
softmax. Those have different spreads, since an expectation over a softmax is pulled toward
the mean and a regression output is not. So a weighted mean is dominated by whichever columns
happen to be widest, and the shared cuts can only undo one global affine map, not 67 of them.

Three fixes, scored with the same document-grouped CV as exp004:

  z      : standardise each column on fit-set mean and sd
  rank   : replace each column by its normal-scored rank, which removes scale and shape
  z-self : standardise each column on each split's own statistics, so a member whose output
           drifts between test and blind gets re-centred on blind

z-self is the interesting one for the blind gap. Every team loses ~1.4 QWK from public test
to blind, and some of that could be member outputs shifting rather than blind being harder.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from slra_ot.paths import REGENERATED  # noqa: E402

# re-runs write here, never over the committed record
OUT_DIR = os.path.join(str(REGENERATED), os.path.basename(HERE))
os.makedirs(OUT_DIR, exist_ok=True)

from slra_ot import combiners as C  # noqa: E402
from slra_ot.members import aligned, doc_structure, pinned_members  # noqa: E402


def zscore(X, mu=None, sd=None):
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) if sd is None else sd
    return (X - mu) / np.maximum(sd, 1e-6), mu, sd


def ranknorm(X):
    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        r = rankdata(X[:, j]) / (len(X) + 1.0)
        out[:, j] = norm.ppf(r)
    return out


def main():
    members = pinned_members(HERE)
    tags = [m["tag"] for m in members]
    test_ids, X, y = aligned(members, "test")
    docs, _ = doc_structure(test_ids, "test")
    print(f"pool {len(tags)} | test {X.shape}\n")

    sd = X.std(0)
    print(f"per-member score sd: min={sd.min():.3f} max={sd.max():.3f} "
          f"ratio={sd.max()/sd.min():.2f}")
    widest = np.argsort(sd)[::-1][:3]
    narrow = np.argsort(sd)[:3]
    print("  widest :", [f"{tags[j]} {sd[j]:.2f}" for j in widest])
    print("  narrow :", [f"{tags[j]} {sd[j]:.2f}" for j in narrow], "\n")

    Xz, _, _ = zscore(X)
    Xr = ranknorm(X)

    rows = []
    for name, Xv in [("raw scores", X), ("z-scored", Xz), ("rank-normalised", Xr)]:
        for cname, factory in [("greedy", lambda: C.GreedyMultiset(rounds=25)),
                               ("bagged greedy", lambda: C.BaggedGreedy(n_bags=15, model_frac=0.5, rounds=15))]:
            q, _ = C.cv_qwk_full(factory, Xv, y, docs, n_folds=5)
            rows.append({"transform": name, "combiner": cname, "cv_qwk": q * 100})
            print(f"{name:16s} {cname:14s} CV={q*100:7.3f}", flush=True)

    rows.sort(key=lambda r: -r["cv_qwk"])
    print(f"\nbest: {rows[0]['transform']} + {rows[0]['combiner']} = {rows[0]['cv_qwk']:.3f}")
    json.dump({"results": rows, "sd_ratio": float(sd.max() / sd.min())},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
