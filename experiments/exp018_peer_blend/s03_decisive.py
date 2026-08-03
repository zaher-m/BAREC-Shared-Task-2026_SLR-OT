"""exp018 step 3: the tie-breaker. Gold copied through, or a peer prediction?

Step 2 left it open, and two facts pull in opposite directions:

  - sents_RL agrees with gold at 94.4% exact / QWK 98.2 over 86,700 gold-matched sentences.
    No model gets that on 19 ordinal classes unless it is scoring its own training data,
    which points at a copy of the human label.
  - QWK(sents_RL, sub07) on the covered blind rows is 91.4, while sub07's real blind QWK is
    85.5. If sents_RL were gold there, sub07 would have to be ~6 points better on the
    covered rows than it is overall.

Both are testable.

Test A, memorisation: split the gold-matched agreement by BAREC split. A model trained on
train would agree strongly on train and fall apart on dev/test. Equal agreement across all
three rules that out.

Test B, is the covered subset just easier? Coverage is not random, it depends on whether the
sentence's source document is in BAREC-10M. So fit P(covered | features) on the blind set and
use it to reweight the test split, where gold is available, toward the covered-like
distribution. If sub07's QWK there goes up by ~6 points, the 91.4 is explained by subset
difficulty. If it stays flat, 91.4 is too high to be agreement with truth and has to be two
correlated models agreeing with each other.
"""
import glob
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
from slra_ot.members import aligned, blind_matrix, pinned_members  # noqa: E402
from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.paths import BAREC10M_ANNOTATIONS, RAW_SPLITS  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

SRC = BAREC10M_ANNOTATIONS
WS = re.compile(r"\s+")


def key(s):
    return WS.sub(" ", unicodedata.normalize("NFKC", str(s)).replace("ـ", "")).strip()


def weighted_qwk(y, p, w):
    """QWK with per-row weights, reweighting both the confusion counts and the marginals."""
    y, p, w = np.asarray(y) - 1, np.asarray(p) - 1, np.asarray(w, float)
    w = w / w.sum() * len(y)
    O = np.zeros((19, 19))
    for a, b, ww in zip(y, p, w):
        O[a, b] += ww
    hy, hp = O.sum(1), O.sum(0)
    j = np.arange(19)
    W = (j[:, None] - j[None, :]) ** 2
    num = float((W * O).sum())
    den = float((W * np.outer(hy, hp)).sum()) / w.sum()
    return 100.0 * (1 - num / den)


def main():
    # ---------- collect sents_RL, grouped by which BAREC split it matches -------
    split_lut, blind_key = {}, {}
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(RAW_SPLITS[split])
        col = "Readability_Level_19" if "Readability_Level_19" in df.columns else df.columns[-1]
        for s, l in zip(df.Sentence, df[col]):
            split_lut[key(s)] = (split, int(l))
    bl = pd.read_parquet(RAW_SPLITS["blind"])
    for i, s in enumerate(bl.Sentence):
        blind_key[key(s)] = i

    files = [f for f in glob.glob(os.path.join(SRC, "**", "*.json"), recursive=True)
             if "MACOSX" not in f]
    per_split = {"train": [], "validation": [], "test": []}
    peer_blind = np.full(len(bl), -1, int)
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        raws, rls = d.get("raw_sents", []), d.get("sents_RL", [])
        for i, raw in enumerate(raws):
            if i >= len(rls):
                break
            lvl = rls[i]
            if not isinstance(lvl, int) or not (1 <= lvl <= 19):
                continue
            k = key(raw)
            if k in split_lut:
                sp, g = split_lut[k]
                per_split[sp].append((g, lvl))
            elif k in blind_key:
                peer_blind[blind_key[k]] = lvl

    print("Test A: sents_RL vs gold, by BAREC split")
    print(f"{'split':>12s} {'n':>8s} {'QWK':>8s} {'exact %':>9s} {'MAE':>7s}")
    testA = {}
    for sp in ["train", "validation", "test"]:
        a = np.array([x for x, _ in per_split[sp]]); b = np.array([y for _, y in per_split[sp]])
        q = fast_qwk(a - 1, b - 1) * 100
        testA[sp] = {"n": len(a), "qwk": float(q), "exact": float((a == b).mean() * 100),
                     "mae": float(np.abs(a - b).mean())}
        print(f"{sp:>12s} {len(a):8d} {q:8.3f} {100*(a==b).mean():9.2f} "
              f"{np.abs(a-b).mean():7.4f}")
    gap = testA["train"]["exact"] - 0.5 * (testA["validation"]["exact"] + testA["test"]["exact"])
    print(f"\ntrain-minus-heldout exact-agreement gap: {gap:+.2f} points")
    print("  a model trained on train would show a gap of tens of points here.")

    # ---------- rebuild sub07 -----------------------------------------------------
    AD = ("_ad", "_ad2", "_ad3", "_ad4")
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    ids, Xt, yt = aligned(members, "test")
    Xb = blind_matrix(tags)
    W = C.GreedyMultiset(rounds=30).fit(Xt, yt).weights_
    s_test, s_blind = Xt @ W, Xb @ W
    th = C.prior_shrunk_thresholds(s_test, yt, k=0.35)
    p_test = labels_from_thresholds(s_test, th)
    p_blind = labels_from_thresholds(s_blind, th)

    cov = peer_blind > 0
    print(f"\nblind coverage {cov.sum()}/{len(cov)} ({100*cov.mean():.1f}%)")
    qcov = fast_qwk(peer_blind[cov] - 1, p_blind[cov] - 1) * 100
    print(f"QWK(sents_RL, sub07) on covered blind rows = {qcov:.2f}   "
          f"(sub07 vs real gold, whole blind set = 85.50)")

    # spread check: a subset with a wider spread gets a higher QWK for free
    print(f"\nour blind score sd:  covered {s_blind[cov].std():.3f}  "
          f"uncovered {s_blind[~cov].std():.3f}  all {s_blind.std():.3f}")
    print(f"blind pred sd :  covered {p_blind[cov].std():.3f}  "
          f"uncovered {p_blind[~cov].std():.3f}  all {p_blind.std():.3f}")

    # ---------- Test B: reweight test toward the covered-like distribution ------
    # features that might drive coverage and can be computed on both splits
    def feats(texts, score):
        nw = np.array([len(str(t).split()) for t in texts], float)
        nc = np.array([len(str(t)) for t in texts], float)
        return np.column_stack([np.log1p(nw), np.log1p(nc), score, score ** 2,
                                nc / np.maximum(nw, 1)])

    # align test text by ID rather than by row order
    te_raw = pd.read_parquet(RAW_SPLITS["test"])
    idcol = [c for c in te_raw.columns if "id" in c.lower()][0]
    tmap = dict(zip(te_raw[idcol].astype(str), te_raw.Sentence))
    te_texts = [tmap[str(i)] for i in ids]

    Fb, Ft = feats(bl.Sentence.values, s_blind), feats(te_texts, s_test)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.vstack([Fb, Ft]))
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Fb), cov.astype(int))
    auc_note = lr.score(sc.transform(Fb), cov.astype(int))
    pc_b = lr.predict_proba(sc.transform(Fb))[:, 1]
    pc_t = lr.predict_proba(sc.transform(Ft))[:, 1]
    print(f"\nTest B: P(covered | length, score) fitted on blind, "
          f"in-sample accuracy {auc_note:.3f}")
    print(f"  mean propensity: blind covered {pc_b[cov].mean():.3f}  "
          f"blind uncovered {pc_b[~cov].mean():.3f}  test rows {pc_t.mean():.3f}")

    base = fast_qwk(yt - 1, p_test - 1) * 100
    # importance weights: move the test distribution toward covered-like
    wcov = pc_t / np.maximum(1 - pc_t, 1e-6)
    wcov = np.clip(wcov / wcov.mean(), 0.02, 50)
    wunc = (1 - pc_t) / np.maximum(pc_t, 1e-6)
    wunc = np.clip(wunc / wunc.mean(), 0.02, 50)
    q_cov_like = weighted_qwk(yt, p_test, wcov)
    q_unc_like = weighted_qwk(yt, p_test, wunc)
    print(f"\nour TEST QWK, unweighted                    {base:8.2f}")
    print(f"TEST QWK, reweighted to covered-like     {q_cov_like:8.2f}  "
          f"({q_cov_like-base:+.2f})")
    print(f"TEST QWK, reweighted to uncovered-like   {q_unc_like:8.2f}  "
          f"({q_unc_like-base:+.2f})")
    print(f"\nrequired for sents_RL-as-gold to hold: about {qcov-85.50:+.2f} on the "
          f"covered rows.")
    print(f"observed subset effect of that kind on test:  {q_cov_like-base:+.2f}")

    verdict = ("GOLD copy-through is consistent with the data"
               if (q_cov_like - base) >= 0.6 * (qcov - 85.50)
               else "sents_RL on blind behaves like a PEER MODEL's prediction, not gold")
    print(f"\nVERDICT: {verdict}")

    json.dump({"testA": testA, "train_minus_heldout_exact_gap": float(gap),
               "coverage": int(cov.sum()), "qwk_peer_vs_sub07_covered": float(qcov),
               "sd_score_covered": float(s_blind[cov].std()),
               "sd_score_uncovered": float(s_blind[~cov].std()),
               "test_qwk": float(base), "test_qwk_covered_like": float(q_cov_like),
               "test_qwk_uncovered_like": float(q_unc_like),
               "required_subset_effect": float(qcov - 85.50),
               "observed_subset_effect": float(q_cov_like - base),
               "verdict": verdict},
              open(os.path.join(OUT_DIR, "s03_decisive.json"), "w"), indent=2)
    np.save(os.path.join(OUT_DIR, "peer_labels_blind.npy"), peer_blind)


if __name__ == "__main__":
    main()
