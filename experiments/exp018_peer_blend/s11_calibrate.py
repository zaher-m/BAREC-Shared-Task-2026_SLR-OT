"""exp018 step 11: skip the model, measure what blending at a given D/v_o actually buys.

Step 10's factor model failed its own validation: 24.5% median error on v_j, and biased in the
direction that matters, since predictors far from the consensus get their variance
over-estimated. The model cannot tell "different and also good" from "different and noisy". The
peer is the most distant column in the matrix, so its c = 1.98 is an upper bound, not an
estimate. Dropped.

So do it without a model. For each of the 48 members in turn:

  - take it out of the pool and refit the ensemble
  - measure D / v_o between the reduced ensemble and the held-out member, which needs no
    labels and can therefore be computed for the peer too
  - blend it back on the same ~56% row mask at w = 0.20 and measure the QWK change against gold

That gives 48 (x, y) pairs where x is observable for the peer and y is the unknown.
Then read off the answer at the peer's own x = D(ensemble, peer) / v_o.

Caveat up front: if the peer's x falls outside the range the members cover, this is
extrapolation. The script says so rather than quietly interpolating.
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
from slra_ot.members import aligned, blind_matrix, pinned_members  # noqa: E402
from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

AD = ("_ad", "_ad2", "_ad3", "_ad4")
WBLEND = 0.20
EXPAND = 1.02


def main():
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    ids, X, y = aligned(members, "test")
    n = len(y)

    ab = np.array([np.polyfit(X[:, j], y, 1) for j in range(len(tags))])
    U = X * ab[:, 0] + ab[:, 1]
    v_solo = np.array([np.var(y - U[:, j]) for j in range(len(tags))])

    rng = np.random.default_rng(0)
    mask = rng.random(n) < 0.563          # same coverage the peer has on blind

    print(f"{len(tags)} members, mask {mask.sum()}/{n} ({100*mask.mean():.1f}%), "
          f"w={WBLEND}, expand={EXPAND}\n")
    print(f"{'member':34s} {'D/v_o':>7s} {'v_j/v_o':>8s} {'base':>8s} {'blended':>8s} {'delta':>7s}")

    rows = []
    for j in range(len(tags)):
        keep = [k for k in range(len(tags)) if k != j]
        Wk = C.GreedyMultiset(rounds=30).fit(X[:, keep], y).weights_
        s = X[:, keep] @ Wk
        a, b = np.polyfit(s, y, 1)
        u = a * s + b
        th = a * C.prior_shrunk_thresholds(s, y, k=0.35) + b
        v_o = float(np.var(y - u))
        q0 = fast_qwk(y - 1, labels_from_thresholds(u, th) - 1) * 100

        p = U[:, j]
        pm = p[mask] - p[mask].mean() + u[mask].mean()
        D = float(np.mean((u[mask] - pm) ** 2))
        u2 = u.copy()
        bl = (1 - WBLEND) * u[mask] + WBLEND * pm
        u2[mask] = bl.mean() + (bl - bl.mean()) * EXPAND
        q1 = fast_qwk(y - 1, labels_from_thresholds(u2, th) - 1) * 100

        rows.append({"tag": tags[j], "D_over_vo": D / v_o, "vj_over_vo": v_solo[j] / v_o,
                     "base": q0, "blended": q1, "delta": q1 - q0})
        print(f"{tags[j]:34s} {D/v_o:7.3f} {v_solo[j]/v_o:8.3f} {q0:8.3f} {q1:8.3f} "
              f"{q1-q0:+7.3f}", flush=True)

    x = np.array([r["D_over_vo"] for r in rows])
    ydel = np.array([r["delta"] for r in rows])
    vr = np.array([r["vj_over_vo"] for r in rows])

    print(f"\nD/v_o across the members: min {x.min():.3f} median {np.median(x):.3f} "
          f"max {x.max():.3f}")
    print(f"delta across the members: min {ydel.min():+.3f} median {np.median(ydel):+.3f} "
          f"max {ydel.max():+.3f}   positive for {int((ydel>0).sum())}/{len(ydel)}")
    r = float(np.corrcoef(x, ydel)[0, 1])
    print(f"correlation between D/v_o and realised delta: {r:+.3f}")
    print(f"correlation between v_j/v_o and realised delta: "
          f"{np.corrcoef(vr, ydel)[0,1]:+.3f}")

    # the peer's own x, measured on blind
    peer = np.load(os.path.join(HERE, "peer_blind_nogold.npy"))
    cov = ~np.isnan(peer)
    Xb = blind_matrix(tags)
    Ub = Xb * ab[:, 0] + ab[:, 1]
    W = C.GreedyMultiset(rounds=30).fit(X, y).weights_
    wl = W * ab[:, 0]; wl = wl / wl.sum()
    ub = Ub @ wl
    pv = peer[cov] - peer[cov].mean() + ub[cov].mean()
    Dpeer = float(np.mean((ub[cov] - pv) ** 2))
    v_o_blind = 2.5587 * (1 - 0.8550) / (1 - 0.8694)
    xpeer = Dpeer / v_o_blind
    print(f"\npeer: D {Dpeer:.4f} / v_o(blind) {v_o_blind:.4f} = {xpeer:.3f}")
    inside = x.min() <= xpeer <= x.max()
    print(f"      inside the range the members span? {'YES' if inside else 'NO, this is extrapolation'}")

    # the nearest members in x, which needs the fewest assumptions
    order = np.argsort(np.abs(x - xpeer))[:8]
    print(f"\nthe 8 members closest to the peer in D/v_o:")
    for j in order:
        print(f"  D/v_o {x[j]:.3f}  v_j/v_o {vr[j]:.3f}  delta {ydel[j]:+.3f}  {rows[j]['tag']}")
    near = ydel[order]
    print(f"\n  their realised deltas: mean {near.mean():+.3f}  median "
          f"{np.median(near):+.3f}  sd {near.std():.3f}  positive {int((near>0).sum())}/8")

    # linear fit, for reference only
    sl, ic = np.polyfit(x, ydel, 1)
    print(f"\nlinear fit delta = {sl:+.4f} * (D/v_o) {ic:+.4f}  ->  at peer x: "
          f"{sl*xpeer+ic:+.3f}")
    print(f"NOTE the peer's v_j/v_o is the unknown; these members' deltas embed THEIR "
          f"qualities\n     ({vr[order].min():.2f}-{vr[order].max():.2f} for the neighbours "
          f"above), so read the spread, not the point.")

    json.dump({"w": WBLEND, "expand": EXPAND, "rows": rows,
               "peer_D": Dpeer, "peer_v_o": v_o_blind, "peer_x": xpeer,
               "inside_range": bool(inside),
               "neighbour_deltas": near.tolist(),
               "neighbour_mean": float(near.mean()), "neighbour_sd": float(near.std()),
               "linear_read": float(sl * xpeer + ic),
               "corr_x_delta": r},
              open(os.path.join(OUT_DIR, "s11_calibrate.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
