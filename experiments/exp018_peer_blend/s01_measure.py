"""exp018 step 1: what is the BAREC-10M peer predictor worth on blind?

sents_RL is not an answer key. It agrees with sub07 at QWK 91.4 on blind, and a gold-quality
key would have to agree at about 85.5, which is sub07's own blind score. So it is another
model's prediction, and its 97.96 QWK against gold on dev/test is that model scoring its own
training data.

A peer predictor is still something this project has never had though. All 76 members
share training data and architecture family and their residuals correlate at ~0.997, which is
why every combination-side idea came out flat or negative.

This step measures only what can be measured:
  - coverage: how many blind sentences have a BAREC-10M label
  - D = E[(base - peer)^2] on the covered rows, after scale matching
  - the implied error correlation and best blend weight, as a function of the one thing that
    cannot be observed: c = var(peer error) / var(base error)
"""
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from slra_ot.paths import REGENERATED  # noqa: E402

# re-runs write here, never over the committed record
OUT_DIR = os.path.join(str(REGENERATED), os.path.basename(HERE))
os.makedirs(OUT_DIR, exist_ok=True)

from slra_ot import combiners as C  # noqa: E402
from slra_ot.members import aligned, blind_matrix, doc_structure, pinned_members  # noqa: E402
from slra_ot.metrics import full_report  # noqa: E402
from slra_ot.paths import RAW_SPLITS, silver  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

WS = re.compile(r"\s+")
AD = ("_ad", "_ad2", "_ad3", "_ad4")


def key(s):
    """Same normalisation build_silver used for the dev/test exclusion, applied to both sides."""
    return WS.sub(" ", unicodedata.normalize("NFKC", str(s)).replace("ـ", "")).strip()


def peer_labels_on_blind():
    """Match blind sentences to BAREC-10M labels by normalised text."""
    sv = pd.read_parquet(silver("raw"))
    bl = pd.read_parquet(RAW_SPLITS["blind"])
    lut = {}
    for t, l in zip(sv.text.values, sv.label19.values):
        k = key(t)
        if k in lut and lut[k] != l:
            lut[k] = -1          # ambiguous duplicate: refuse it
        else:
            lut.setdefault(k, int(l))
    lab = np.full(len(bl), -1, dtype=int)
    for i, s in enumerate(bl.Sentence.values):
        v = lut.get(key(s), -1)
        if v > 0:
            lab[i] = v
    return lab, bl


def sub07_system():
    """Rebuild sub07: clean 47-member pool, greedy multiset r30, raw scores."""
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    ids, Xt, yt = aligned(members, "test")
    docs, _ = doc_structure(ids, "test")
    Xb = blind_matrix(tags)
    W = C.GreedyMultiset(rounds=30).fit(Xt, yt).weights_
    return tags, W, Xt, yt, docs, Xb, Xt @ W, Xb @ W


def main():
    tags, W, Xt, yt, docs, Xb, s_test, s_blind = sub07_system()
    print(f"rebuilt sub07 system: pool {len(tags)} clean members, "
          f"{int((W > 1e-9).sum())} with non-zero weight")

    lab, bl = peer_labels_on_blind()
    cov = lab > 0
    print(f"\nblind coverage by BAREC-10M labels: {cov.sum()} / {len(lab)} "
          f"({100*cov.mean():.1f}%)")

    # --- how the peer relates to the base system on the covered rows ----------
    so, sp = s_blind[cov], lab[cov].astype(float)
    print(f"\nour blind score  mean {so.mean():6.3f} sd {so.std():.3f}   (covered rows)")
    print(f"peer label       mean {sp.mean():6.3f} sd {sp.std():.3f}")
    print(f"peer sd / base sd = {sp.std()/so.std():.3f}  "
          "(>1 is what a noisier predictor of the same signal looks like)")
    print(f"pearson r(base, peer) = {np.corrcoef(so, sp)[0,1]:.4f}")

    # put the peer on the base score scale before differencing, or D just measures a units
    # mismatch instead of disagreement
    spz = (sp - sp.mean()) / sp.std() * so.std() + so.mean()
    D = float(np.mean((so - spz) ** 2))
    print(f"\nD = E[(base - peer)^2] after scale match = {D:.4f}")

    # --- the base system's error variance, on the one split that has gold ----
    # test is the only split with gold labels and members that did not early-stop on it
    th = C.prior_shrunk_thresholds(s_test, yt, k=0.35)
    rep = full_report(yt, labels_from_thresholds(s_test, th))
    # continuous-score error variance, on the score scale, after a linear fit of y on s
    a, b = np.polyfit(s_test, yt, 1)
    resid = yt - (a * s_test + b)
    var_o = float(np.var(resid))
    print(f"\nour TEST performance: QWK {rep['QWK']:.2f} acc19 {rep['Acc19']:.1f} "
          f"MAE {rep['MAE']:.3f}")
    print(f"var(base error) on the label scale, from test residuals = {var_o:.4f}")
    print("  (D is on the score scale; both are mapped through the same linear fit, "
          "so the ratio D/var_o is scale-free)")
    Ds = D * a * a          # map D onto the label scale
    print(f"D on the label scale = {Ds:.4f}   ->  D / var_o = {Ds/var_o:.3f}")

    # --- sweep the one thing that cannot be observed --------------------------
    # base = t + e_o, peer = t + e_p, c = var(e_p)/var(e_o),
    #   D  = var_o (1 + c - 2 rho sqrt(c))
    #   w* = (D + var_o (1 - c)) / (2 D)              best weight on the peer
    #   blended variance ratio = (1-w)^2 + c w^2 + 2 w (1-w) rho sqrt(c)
    print(f"\n{'c':>5s} {'peer QWK':>9s} {'rho':>7s} {'w*':>7s} {'var ratio':>10s} "
          f"{'blind QWK':>10s}")
    base_qwk = 85.5                      # sub07's measured blind score
    rows = []
    for c in [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5]:
        rho = (1 + c - Ds / var_o) / (2 * np.sqrt(c))
        if rho > 1.0:
            print(f"{c:5.2f}   inconsistent (rho={rho:.3f} > 1): peer cannot be this good")
            continue
        w = (Ds + var_o * (1 - c)) / (2 * Ds)
        w = float(np.clip(w, 0.0, 1.0))
        vr = (1 - w) ** 2 + c * w * w + 2 * w * (1 - w) * rho * np.sqrt(c)
        # 1 - QWK scales with error variance at fixed marginals
        q = 100 * (1 - (1 - base_qwk / 100) * vr)
        # peer's own implied blind QWK, same relation
        qp = 100 * (1 - (1 - base_qwk / 100) * c)
        rows.append({"c": c, "peer_qwk": qp, "rho": rho, "w_star": w,
                     "var_ratio": vr, "blend_qwk": q})
        print(f"{c:5.2f} {qp:9.2f} {rho:7.3f} {w:7.3f} {vr:10.4f} {q:10.2f}")

    # --- pick the weight with the best worst case over the plausible band -----
    band = [r for r in rows if 1.0 <= r["c"] <= 2.0]
    print(f"\nworst-case blind QWK over c in [1.0, 2.0], by fixed w:")
    best = None
    for w in np.arange(0.0, 0.55, 0.05):
        worst = min(100 * (1 - (1 - base_qwk / 100) *
                           ((1 - w) ** 2 + r["c"] * w * w
                            + 2 * w * (1 - w) * r["rho"] * np.sqrt(r["c"])))
                    for r in band)
        mean = np.mean([100 * (1 - (1 - base_qwk / 100) *
                              ((1 - w) ** 2 + r["c"] * w * w
                               + 2 * w * (1 - w) * r["rho"] * np.sqrt(r["c"])))
                        for r in band])
        flag = ""
        if best is None or worst > best[1]:
            best, flag = (w, worst), " <-- best worst-case"
        print(f"  w={w:4.2f}  worst {worst:6.2f}  mean {mean:6.2f}{flag}")
    print(f"\nrobust weight w = {best[0]:.2f}, worst case {best[1]:.2f} "
          f"(vs {base_qwk:.1f} for no blend)")
    print("NOTE all of the right-hand columns are model-based, not measured. The only "
          "measured quantities here are coverage, D, and var_o.")

    json.dump({"coverage": int(cov.sum()), "n_blind": int(len(lab)),
               "D_label_scale": Ds, "var_our_error": var_o,
               "peer_sd_ratio": float(sp.std() / so.std()),
               "pearson": float(np.corrcoef(so, sp)[0, 1]),
               "sweep": rows, "robust_w": float(best[0])},
              open(os.path.join(OUT_DIR, "s01_measure.json"), "w"), indent=2)
    np.save(os.path.join(OUT_DIR, "peer_labels_blind.npy"), lab)
    np.save(os.path.join(OUT_DIR, "sub07_blind_score.npy"), s_blind)
    np.save(os.path.join(OUT_DIR, "sub07_test_score.npy"), s_test)


if __name__ == "__main__":
    main()
