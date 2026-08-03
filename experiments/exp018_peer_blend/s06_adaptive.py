"""exp018 step 6: flat peer weight, or weight it by the ensemble's own uncertainty?

The idea: an outside predictor should be worth most on rows where the ensemble is unsure,
and worth nothing where all 48 members already agree. So use a per-row weight

    w_i = w0 * s_i / mean(s)        s_i = sd across the members on row i

which concentrates the peer's influence on the rows the pool disagrees about.

Not the same as exp013, which shrank uncertain rows toward the global mean (+0.02, and not
consistent). That throws information away; this replaces it with an independent estimate.

Scored by document-grouped 5-fold CV with the combiner, the label-scale map and the cuts all
refitted per fold, using the stand-in peer from step 5.
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
from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.thresholds import FastQWKThresholds, labels_from_thresholds  # noqa: E402

AD = ("_ad", "_ad2", "_ad3", "_ad4")
COVERAGE = 0.529


def blend_rows(u, peer, have, wrow, expand=1.0):
    out = u.copy()
    if not have.any():
        return out
    p = peer[have] - peer[have].mean() + u[have].mean()
    w = wrow[have]
    b = (1 - w) * u[have] + w * p
    if expand != 1.0:
        b = b.mean() + (b - b.mean()) * expand
    out[have] = b
    return out


def main():
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    ids, X, y = aligned(members, "test")
    docs, _ = doc_structure(ids, "test")

    solo = [(FastQWKThresholds(rounds=6).fit(X[:, j], y).best_qwk_ * 100, j)
            for j in range(len(tags))]
    solo.sort(reverse=True)
    jh = solo[0][1]
    keep = [j for j in range(len(tags)) if j != jh]
    Xk, peer_raw = X[:, keep], X[:, jh]
    print(f"synthetic peer {tags[jh]}, pool {Xk.shape[1]} members")

    rng = np.random.default_rng(0)
    have = rng.random(len(y)) < COVERAGE
    # per-row internal disagreement, from the members that carry weight
    Wfull = C.GreedyMultiset(rounds=30).fit(Xk, y).weights_
    used = Wfull > 1e-9
    disp = Xk[:, used].std(axis=1)
    print(f"per-row member sd: mean {disp.mean():.3f} "
          f"p10 {np.percentile(disp,10):.3f} p90 {np.percentile(disp,90):.3f}")

    fold = C.doc_folds(docs, 5, 0)

    def cv(mode, w0, expand=1.0, cap=0.6):
        pred = np.empty(len(y), int)
        for f in range(5):
            tr, te = fold != f, fold == f
            Wf = C.GreedyMultiset(rounds=30).fit(Xk[tr], y[tr]).weights_
            sf_tr, sf_te = Xk[tr] @ Wf, Xk[te] @ Wf
            af, bf = np.polyfit(sf_tr, y[tr], 1)
            thf = af * C.prior_shrunk_thresholds(sf_tr, y[tr], k=0.35) + bf
            apf, bpf = np.polyfit(peer_raw[tr], y[tr], 1)
            u_te = af * sf_te + bf
            p_te = apf * peer_raw[te] + bpf
            uf = Wf > 1e-9
            d_tr, d_te = Xk[tr][:, uf].std(1), Xk[te][:, uf].std(1)
            if mode == "flat":
                wrow = np.full(te.sum(), w0)
            elif mode == "prop":
                wrow = np.clip(w0 * d_te / d_tr.mean(), 0.0, cap)
            elif mode == "rank":
                r = np.searchsorted(np.sort(d_tr), d_te) / max(len(d_tr), 1)
                wrow = np.clip(2 * w0 * r, 0.0, cap)
            pred[te] = labels_from_thresholds(
                blend_rows(u_te, p_te, have[te], wrow, expand), thf)
        return fast_qwk(y - 1, pred - 1) * 100

    base = cv("flat", 0.0)
    print(f"\nbase (no peer)                    CV QWK {base:.3f}")
    print(f"\n{'w0':>6s} {'flat':>9s} {'prop':>9s} {'rank':>9s}")
    rows = []
    for w0 in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        a, b, c = cv("flat", w0), cv("prop", w0), cv("rank", w0)
        rows.append({"w0": w0, "flat": a, "prop": b, "rank": c})
        print(f"{w0:6.2f} {a:9.3f} {b:9.3f} {c:9.3f}")

    print(f"\nwith expansion 1.02")
    print(f"{'w0':>6s} {'flat':>9s} {'prop':>9s}")
    rows2 = []
    for w0 in [0.15, 0.20, 0.25, 0.30]:
        a, b = cv("flat", w0, 1.02), cv("prop", w0, 1.02)
        rows2.append({"w0": w0, "flat_e": a, "prop_e": b})
        print(f"{w0:6.2f} {a:9.3f} {b:9.3f}")

    allr = [("flat", r["w0"], r["flat"]) for r in rows] + \
           [("prop", r["w0"], r["prop"]) for r in rows] + \
           [("rank", r["w0"], r["rank"]) for r in rows] + \
           [("flat+e", r["w0"], r["flat_e"]) for r in rows2] + \
           [("prop+e", r["w0"], r["prop_e"]) for r in rows2]
    allr.sort(key=lambda t: -t[2])
    print(f"\nbest 5 by CV:")
    for m, w, q in allr[:5]:
        print(f"  {m:8s} w0={w:.2f}  {q:.3f}  ({q-base:+.3f})")
    best = allr[0]
    print(f"\nVERDICT: {'adaptive beats flat' if 'prop' in best[0] or 'rank' in best[0] else 'flat weighting wins, keep it simple'}")

    json.dump({"base_cv": base, "rows": rows, "rows_expand": rows2,
               "best": {"mode": best[0], "w0": best[1], "cv": best[2]}},
              open(os.path.join(OUT_DIR, "s06_adaptive.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
