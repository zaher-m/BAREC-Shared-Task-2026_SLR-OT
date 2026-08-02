"""exp004: try to claw back the 0.75 QWK of selection overfit on the bigger pool.

Where things stood: sub02 (45 members, plain greedy, single threshold fit) scored 85.2 on
blind against a CV estimate of 86.623, a -1.4 shift. The v5 system showed the same shift
(86.398 -> 85.00) and so did thylinao (86.30 -> 84.80), which made it look like CV on test
was a calibrated predictor of blind and that beating 85.3 needed CV >= ~86.75.

Two things to spend: the pool had grown from 45 to 67 members, and exp003 had measured a 0.75
gap between the in-sample number (87.369) and CV (86.623). That gap is selection variance,
and bagging is the usual fix.

Everything is scored by document-grouped 5-fold CV on test with the combiner and the cuts
refitted inside each fold.

Later note: the constant-offset assumption was wrong. The offset depends on how much
contaminated weight the system carries (exp010, exp017).
"""
import json
import os
import sys
import time

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


def main():
    members = pinned_members(HERE)
    tags = [m["tag"] for m in members]
    test_ids, X, y = aligned(members, "test")
    docs, _ = doc_structure(test_ids, "test")
    print(f"pool: {len(tags)} members | test {X.shape} | {len(np.unique(docs))} documents\n")

    # the 45-member pool exp003 shipped, as a like-for-like reference
    p45 = [i for i, t in enumerate(tags)
           if t in set(json.load(open(os.path.join(ROOT, "experiments",
                                                   "exp003_production_ensemble",
                                                   "results.json")))["production"]["members"])
           or not t.endswith("_ad")]

    cands = [
        ("greedy [45-pool ref]", lambda: C.GreedyMultiset(rounds=25), "single", p45),
        ("greedy", lambda: C.GreedyMultiset(rounds=25), "single", None),
        ("greedy + bagged thr", lambda: C.GreedyMultiset(rounds=25), "bagged", None),
        ("topk-20 uniform", lambda: C.TopKUniform(k=20), "single", None),
        ("topk-30 uniform", lambda: C.TopKUniform(k=30), "single", None),
        ("bagged greedy m.5", lambda: C.BaggedGreedy(n_bags=15, model_frac=0.5, rounds=15), "single", None),
        ("bagged greedy m.35", lambda: C.BaggedGreedy(n_bags=15, model_frac=0.35, rounds=15), "single", None),
        ("bagged greedy m.5 + rows.8", lambda: C.BaggedGreedy(n_bags=15, model_frac=0.5, rounds=15, row_frac=0.8), "single", None),
        ("bagged greedy m.5 + bagged thr", lambda: C.BaggedGreedy(n_bags=15, model_frac=0.5, rounds=15), "bagged", None),
    ]

    rows = []
    for name, factory, thr, cols in cands:
        Xc = X if cols is None else X[:, cols]
        t0 = time.time()
        q, _ = C.cv_qwk_full(factory, Xc, y, docs, n_folds=5, thresholds=thr)
        rows.append({"method": name, "n_members": Xc.shape[1], "thresholds": thr,
                     "cv_qwk": q * 100, "secs": round(time.time() - t0)})
        print(f"{name:32s} n={Xc.shape[1]:3d} thr={thr:6s} CV={q*100:7.3f}  ({time.time()-t0:.0f}s)",
              flush=True)

    rows.sort(key=lambda r: -r["cv_qwk"])
    ref = next(r for r in rows if r["method"].startswith("greedy [45"))
    print(f"\nbest: {rows[0]['method']}  CV={rows[0]['cv_qwk']:.3f}")
    print(f"  vs 45-pool greedy reference {ref['cv_qwk']:.3f}  -> {rows[0]['cv_qwk']-ref['cv_qwk']:+.3f}")
    print(f"  implied blind (CV - 1.40)   {rows[0]['cv_qwk']-1.40:.2f}   (need > 85.3)")

    json.dump({"n_pool": len(tags), "tags": tags, "results": rows},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
