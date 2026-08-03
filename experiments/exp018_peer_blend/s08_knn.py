"""exp018 step 8: does near-duplicate retrieval extend the override from step 4?

Step 4 needed verbatim text and only found 68 blind rows (0.84%). BAREC documents repeat
phrasing a lot, same genres and textbooks and authors, so relaxing "identical" to "very
similar" should cover more. The question is whether the label precision survives it, because a
wrong override is worse than none: the ensemble is already within +-1 on 71% of rows.

Retrieval is character 3-5 gram TF-IDF with cosine similarity against the public gold
sentences. Character n-grams rather than words because Arabic morphology makes word overlap
brittle, and the d3tok members already cover the morphological view.

Everything is measured on public test with an index built from train+validation only, so the
precision-per-similarity-band curve is out of sample. The blind set then gets an index over
train+validation+test, all public, at whatever threshold that curve supports.
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
from slra_ot.members import aligned, pinned_members  # noqa: E402
from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.paths import RAW_SPLITS  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

WS = re.compile(r"\s+")
AD = ("_ad", "_ad2", "_ad3", "_ad4")


def key(s):
    return WS.sub(" ", unicodedata.normalize("NFKC", str(s)).replace("ـ", "")).strip()


def load_split(split):
    df = pd.read_parquet(RAW_SPLITS[split])
    col = [c for c in df.columns if "19" in c][0]
    idc = [c for c in df.columns if "id" in c.lower()][0]
    return df[idc].astype(str).values, df.Sentence.values, df[col].astype(int).values


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    # ---- the base system on public test, as the reference -------------------
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    ids, Xt, yt = aligned(members, "test")
    W = C.GreedyMultiset(rounds=30).fit(Xt, yt).weights_
    s_test = Xt @ W
    th = C.prior_shrunk_thresholds(s_test, yt, k=0.35)
    p_test = labels_from_thresholds(s_test, th)
    q0 = fast_qwk(yt - 1, p_test - 1) * 100
    print(f"base public-test QWK {q0:.3f}")

    tr_id, tr_tx, tr_y = load_split("train")
    dv_id, dv_tx, dv_y = load_split("validation")
    idx_tx = np.concatenate([tr_tx, dv_tx])
    idx_y = np.concatenate([tr_y, dv_y])
    print(f"index: {len(idx_tx)} public train+dev sentences")

    te_raw = pd.read_parquet(RAW_SPLITS["test"])
    idc = [c for c in te_raw.columns if "id" in c.lower()][0]
    tmap = dict(zip(te_raw[idc].astype(str), te_raw.Sentence))
    te_tx = np.array([tmap[str(i)] for i in ids])

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          max_features=400000, sublinear_tf=True)
    A = vec.fit_transform([key(t) for t in idx_tx])
    B = vec.transform([key(t) for t in te_tx])
    nn = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute").fit(A)
    dist, ind = nn.kneighbors(B)
    sim = 1.0 - dist
    print(f"nearest-neighbour similarity on public test: "
          f"p50 {np.percentile(sim[:,0],50):.3f} p90 {np.percentile(sim[:,0],90):.3f} "
          f"p99 {np.percentile(sim[:,0],99):.3f}")

    # ---- precision per similarity band, and what overriding each band does ---
    print(f"\n{'sim >=':>7s} {'rows':>6s} {'%':>6s} {'nn exact':>9s} {'nn MAE':>8s} "
          f"{'base MAE':>9s} {'QWK hard':>9s} {'QWK a=.5':>9s}")
    rows = []
    for t in [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60]:
        m = sim[:, 0] >= t
        if m.sum() < 5:
            continue
        lab = idx_y[ind[m, 0]]
        hard = p_test.copy(); hard[m] = lab
        qh = fast_qwk(yt - 1, hard - 1) * 100
        a, b0 = np.polyfit(s_test, yt, 1)
        sm = s_test.copy(); sm[m] = 0.5 * s_test[m] + 0.5 * ((lab - b0) / a)
        qs = fast_qwk(yt - 1, labels_from_thresholds(sm, th) - 1) * 100
        rows.append({"thr": t, "n": int(m.sum()), "frac": float(m.mean()),
                     "nn_exact": float((lab == yt[m]).mean() * 100),
                     "nn_mae": float(np.abs(lab - yt[m]).mean()),
                     "base_mae": float(np.abs(p_test[m] - yt[m]).mean()),
                     "qwk_hard": float(qh), "qwk_soft": float(qs)})
        print(f"{t:7.2f} {m.sum():6d} {100*m.mean():6.2f} {100*(lab==yt[m]).mean():9.1f} "
              f"{np.abs(lab-yt[m]).mean():8.3f} {np.abs(p_test[m]-yt[m]).mean():9.3f} "
              f"{qh:9.3f} {qs:9.3f}")

    # weighted 5-NN, blended in with a per-row weight instead of a hard cutoff
    print(f"\nweighted 5-NN, per-row weight = sim0^p, no hard threshold")
    a, b0 = np.polyfit(s_test, yt, 1)
    best = None
    for p in (4, 6, 8, 12):
        for scale in (0.5, 0.75, 1.0):
            wgt = sim / np.maximum(sim.sum(1, keepdims=True), 1e-9)
            knn_lab = (wgt * idx_y[ind]).sum(1)
            wrow = np.clip(scale * sim[:, 0] ** p, 0, 0.95)
            sm = (1 - wrow) * s_test + wrow * ((knn_lab - b0) / a)
            q = fast_qwk(yt - 1, labels_from_thresholds(sm, th) - 1) * 100
            if best is None or q > best[2]:
                best = (p, scale, q)
            print(f"  p={p:2d} scale={scale:4.2f}  mean w {wrow.mean():.4f}  "
                  f"TEST QWK {q:.3f}  ({q-q0:+.3f})")
    print(f"\nbest weighted-kNN: p={best[0]} scale={best[1]:.2f} -> {best[2]:.3f} "
          f"({best[2]-q0:+.3f})")

    hard_best = max(rows, key=lambda r: r["qwk_hard"])
    print(f"best hard threshold: sim>={hard_best['thr']:.2f} "
          f"({hard_best['n']} rows) -> {hard_best['qwk_hard']:.3f} "
          f"({hard_best['qwk_hard']-q0:+.3f})")

    json.dump({"base_test_qwk": q0, "bands": rows,
               "best_weighted": {"p": best[0], "scale": best[1], "qwk": best[2],
                                 "gain": best[2] - q0},
               "best_hard": hard_best}, open(os.path.join(OUT_DIR, "s08_knn.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
