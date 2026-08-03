"""exp018 step 9: why sents_RL must not go into the pool as a member.

The obvious thing to do with a peer predictor is drop it in the pool and let greedy decide
what it is worth. That would be very wrong here.

On the calibration split, which is the public test set, sents_RL is the gold label: 95.66%
exact, QWK 98.3 (step 3). So a combiner selecting on that split sees a member with almost
perfect skill and hands it most of the weight. On blind the same field is an ordinary model
prediction. The ensemble would end up built on a column that behaves completely differently
between the split it was chosen on and the split it is scored on.

This script measures how much weight greedy actually gives it, so the failure is on record as
a number rather than an argument.
"""
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

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
from slra_ot.members import aligned, doc_structure, pinned_members  # noqa: E402
from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.paths import BAREC10M_ANNOTATIONS, RAW_SPLITS  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

SRC = BAREC10M_ANNOTATIONS
WS = re.compile(r"\s+")
AD = ("_ad", "_ad2", "_ad3", "_ad4")


def key(s):
    return WS.sub(" ", unicodedata.normalize("NFKC", str(s)).replace("ـ", "")).strip()


def main():
    import glob
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    ids, X, y = aligned(members, "test")
    docs, _ = doc_structure(ids, "test")

    # sents_RL on the public test split, aligned to the pool's rows
    te = pd.read_parquet(RAW_SPLITS["test"])
    idc = [c for c in te.columns if "id" in c.lower()][0]
    tmap = dict(zip(te[idc].astype(str), te.Sentence))
    want = defaultdict(list)
    for r, i in enumerate(ids):
        want[key(tmap[str(i)])].append(r)
    acc = defaultdict(list)
    for f in glob.glob(os.path.join(SRC, "**", "*.json"), recursive=True):
        if "MACOSX" in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        raws, rls = d.get("raw_sents", []), d.get("sents_RL", [])
        for j, raw in enumerate(raws):
            if j >= len(rls):
                break
            lvl = rls[j]
            if isinstance(lvl, int) and 1 <= lvl <= 19:
                k = key(raw)
                if k in want:
                    acc[k].append(lvl)
    peer_te = np.full(len(y), np.nan)
    for k, v in acc.items():
        for r in want[k]:
            peer_te[r] = float(np.mean(v))
    have = ~np.isnan(peer_te)
    # rows with no match fall back to the pool mean, the most neutral filler available
    col = np.where(have, peer_te, np.nan)
    col[~have] = X.mean(1)[~have]
    print(f"sents_RL available on {have.sum()}/{len(y)} public-test rows "
          f"({100*have.mean():.1f}%)")
    print(f"  on those rows it agrees with gold at QWK "
          f"{fast_qwk(y[have]-1, np.rint(peer_te[have]).astype(int)-1)*100:.2f}, "
          f"exact {100*(np.rint(peer_te[have])==y[have]).mean():.2f}%")

    Xp = np.column_stack([X, col])
    tags_p = tags + ["BAREC10M_sentsRL"]

    print(f"\ngreedy multiset over the clean pool, WITHOUT the peer column")
    W0 = C.GreedyMultiset(rounds=30).fit(X, y).weights_
    s0 = X @ W0
    th0 = C.prior_shrunk_thresholds(s0, y, k=0.35)
    q0 = fast_qwk(y - 1, labels_from_thresholds(s0, th0) - 1) * 100
    print(f"  in-sample TEST QWK {q0:.3f}")

    print(f"\ngreedy multiset over the same pool WITH the peer column added")
    W1 = C.GreedyMultiset(rounds=30).fit(Xp, y).weights_
    s1 = Xp @ W1
    th1 = C.prior_shrunk_thresholds(s1, y, k=0.35)
    q1 = fast_qwk(y - 1, labels_from_thresholds(s1, th1) - 1) * 100
    wp = float(W1[-1])
    print(f"  in-sample TEST QWK {q1:.3f}  ({q1-q0:+.3f})")
    print(f"  weight handed to BAREC10M_sentsRL: {100*wp:.1f}%")
    top = sorted(((float(W1[j]), tags_p[j]) for j in range(len(tags_p)) if W1[j] > 1e-9),
                 reverse=True)[:6]
    for wv, t in top:
        print(f"    {100*wv:5.1f}%  {t}")

    print(f"\ndocument-grouped 5-fold CV (which cannot see the problem either, because "
          f"every fold is inside the same contaminated split):")
    for label, M, tg in (("without peer", X, tags), ("with peer", Xp, tags_p)):
        fold = C.doc_folds(docs, 5, 0)
        pred = np.empty(len(y), int)
        for f in range(5):
            tr, te_ = fold != f, fold == f
            c = C.GreedyMultiset(rounds=30).fit(M[tr], y[tr])
            th = C.prior_shrunk_thresholds(c.score(M[tr]), y[tr], k=0.35)
            pred[te_] = labels_from_thresholds(c.score(M[te_]), th)
        print(f"  {label:14s} CV QWK {fast_qwk(y-1, pred-1)*100:.3f}")

    print(f"\nCONCLUSION: the combiner puts {100*wp:.1f}% of the weight on a column that is "
          f"the gold label on this split\nand an ordinary model prediction on blind. Both the "
          f"in-sample number and the grouped CV\nnumber endorse it, because both are computed "
          f"on the split where the column is gold.\nYou have to know where the column came "
          f"of the column. No cross-validation\nwithin the calibration split can "
          f"detect it. Hence the post-hoc blend in build_final.py,\nwhere the peer's weight is "
          f"set by an external variance argument and never fitted.")

    json.dump({"peer_rows_on_test": int(have.sum()),
               "peer_qwk_vs_gold_on_test": float(
                   fast_qwk(y[have]-1, np.rint(peer_te[have]).astype(int)-1)*100),
               "greedy_weight_on_peer": wp, "insample_without": q0, "insample_with": q1},
              open(os.path.join(OUT_DIR, "s09_trap.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
