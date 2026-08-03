"""exp018 step 12: redo step 11 with a measurement that matches the blind setup.

Step 11 measured the blend gain against a baseline fitted and evaluated on the same test rows.
That baseline is in-sample optimal, so any fixed-weight change to it looks bad, which biases
the whole thing against blending. On blind the weights were fitted on test and applied to
different rows, i.e. out of sample. The dry run showed the bias is big enough to flip the sign:
for arabertv2_wkl_slvB, the test fit gave -0.016 and grouped CV gave +0.065 at the same w.

So run the sweep again with document-grouped 5-fold CV, refitting the combiner, the
score-to-label map and the cuts inside each fold. Eleven members instead of all 48, since each
point now costs five greedy fits. They span D/v_o from 0.098 to 0.612 and the peer sits at
0.540, inside that range.

Read the answer off the members nearest the peer. If they are negative here too, the blend does
not help and sub10 should drop channel B.
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
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

AD = ("_ad", "_ad2", "_ad3", "_ad4")
WS = [0.10, 0.20, 0.30]
EXPAND = 1.02
PICK = ["arabertv2_soft_slv", "arabertv2_wkl_slvlr1", "arabertv2_large_emd_slv",
        "arabertv2_corn_s2", "araelectra_soft_d3", "arbertv2_soft_d3",
        "camelbert_wkl_d3", "aramodern_soft_raw", "marbert_corn_d3",
        "aramodern_reg_d3", "arabertv2_large_reg_raw"]


def main():
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    ids, X, y = aligned(members, "test")
    docs, _ = doc_structure(ids, "test")
    fold = C.doc_folds(docs, 5, 0)
    rng = np.random.default_rng(0)
    mask = rng.random(len(y)) < 0.563

    prev = {r["tag"]: r for r in json.load(open(os.path.join(HERE, "s11_calibrate.json")))["rows"]}
    print(f"{len(PICK)} members, grouped 5-fold CV, mask {100*mask.mean():.1f}%, "
          f"expand {EXPAND}\n")
    print(f"{'member':30s} {'D/v_o':>6s} {'vj/vo':>6s} {'base':>8s} "
          + " ".join(f"{'w='+format(w,'.2f'):>9s}" for w in WS)
          + f" {'best':>7s} {'fitΔ':>7s}", flush=True)

    out = []
    for tag in PICK:
        j = tags.index(tag)
        keep = [k for k in range(len(tags)) if k != j]
        pred = {w: np.empty(len(y), int) for w in [0.0] + WS}
        for f in range(5):
            tr, te = fold != f, fold == f
            Wf = C.GreedyMultiset(rounds=30).fit(X[tr][:, keep], y[tr]).weights_
            s_tr, s_te = X[tr][:, keep] @ Wf, X[te][:, keep] @ Wf
            a, b = np.polyfit(s_tr, y[tr], 1)
            th = a * C.prior_shrunk_thresholds(s_tr, y[tr], k=0.35) + b
            u_te = a * s_te + b
            ap, bp = np.polyfit(X[tr][:, j], y[tr], 1)
            p_te = ap * X[te][:, j] + bp
            mk = mask[te]
            for w in [0.0] + WS:
                uu = u_te.copy()
                if w > 0 and mk.any():
                    pm = p_te[mk] - p_te[mk].mean() + u_te[mk].mean()
                    bb = (1 - w) * u_te[mk] + w * pm
                    uu[mk] = bb.mean() + (bb - bb.mean()) * EXPAND
                pred[w][te] = labels_from_thresholds(uu, th)
        q = {w: fast_qwk(y - 1, pred[w] - 1) * 100 for w in [0.0] + WS}
        d = {w: q[w] - q[0.0] for w in WS}
        bw = max(WS, key=lambda w: d[w])
        out.append({"tag": tag, "D_over_vo": prev[tag]["D_over_vo"],
                    "vj_over_vo": prev[tag]["vj_over_vo"], "cv_base": q[0.0],
                    "cv_delta": {str(w): d[w] for w in WS}, "best_w": bw,
                    "best_delta": d[bw], "fit_delta": prev[tag]["delta"]})
        print(f"{tag:30s} {prev[tag]['D_over_vo']:6.3f} {prev[tag]['vj_over_vo']:6.3f} "
              f"{q[0.0]:8.3f} " + " ".join(f"{d[w]:+9.3f}" for w in WS)
              + f" {d[bw]:+7.3f} {prev[tag]['delta']:+7.3f}", flush=True)

    x = np.array([r["D_over_vo"] for r in out])
    print(f"\ninstrument bias (CV minus test-fit) at w=0.20: "
          f"mean {np.mean([r['cv_delta']['0.2'] - r['fit_delta'] for r in out]):+.3f}")
    for w in WS:
        dd = np.array([r["cv_delta"][str(w)] for r in out])
        print(f"  w={w:.2f}: CV delta mean {dd.mean():+.3f}  median {np.median(dd):+.3f}  "
              f"positive {int((dd>0).sum())}/{len(dd)}  "
              f"corr with D/v_o {np.corrcoef(x, dd)[0,1]:+.3f}")

    xpeer = 0.540
    near = sorted(out, key=lambda r: abs(r["D_over_vo"] - xpeer))[:4]
    print(f"\nthe 4 members nearest the peer's D/v_o = {xpeer}:")
    for r in near:
        print(f"  D/v_o {r['D_over_vo']:.3f}  vj/vo {r['vj_over_vo']:.3f}  "
              + "  ".join(f"w={w}:{r['cv_delta'][str(w)]:+.3f}" for w in WS)
              + f"   {r['tag']}")
    for w in WS:
        nd = np.array([r["cv_delta"][str(w)] for r in near])
        print(f"  neighbours at w={w:.2f}: mean {nd.mean():+.3f}  positive "
              f"{int((nd>0).sum())}/{len(nd)}")

    best_overall = max(WS, key=lambda w: np.mean([r["cv_delta"][str(w)] for r in
                                                  sorted(out, key=lambda r: abs(r["D_over_vo"]-xpeer))[:4]]))
    nd = np.mean([r["cv_delta"][str(best_overall)] for r in near])
    verdict = ("KEEP channel B" if nd > 0.02 else
               "DROP channel B, the blend does not help under the better measurement")
    print(f"\nVERDICT: {verdict}  (best w {best_overall}, neighbour mean {nd:+.3f})")

    json.dump({"w_grid": WS, "expand": EXPAND, "rows": out, "peer_x": xpeer,
               "neighbour_mean_by_w": {str(w): float(np.mean(
                   [r["cv_delta"][str(w)] for r in near])) for w in WS},
               "verdict": verdict},
              open(os.path.join(OUT_DIR, "s12_calibrate_cv.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
