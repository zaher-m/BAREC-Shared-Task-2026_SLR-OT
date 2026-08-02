"""exp008: tune the combiner on the clean 66-member pool, and check what sub03 shipped.

Three loose ends from exp004/005/006 to close before the final submission:

  - exp004 had bagged greedy at model_frac 0.35 (86.839) beating 0.50 (86.811), but that was
    on the pool that still had the collapsed member in it. Re-run it clean.
  - sub03 averages the combiner over 3 bagging seeds, but CV only ever measured one seed, so
    the number the submission was chosen on has not actually been verified.
  - averaging different combiners (greedy + bagged + NNLS) has not been tried and should be
    cheaper variance reduction than any of them alone.

Same protocol throughout: document-grouped 5-fold CV on test, combiner and cuts refitted in
every fold.
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


class MultiSeedBagged:
    """Bagged greedy, averaged over several bagging seeds."""

    def __init__(self, n_seeds=3, **kw):
        self.n_seeds, self.kw = n_seeds, kw

    def fit(self, X, y):
        W = np.zeros(X.shape[1])
        for s in range(self.n_seeds):
            W += C.BaggedGreedy(seed=s, **self.kw).fit(X, y).weights_
        self.weights_ = W / self.n_seeds
        return self

    def score(self, X):
        return X @ self.weights_


class BlendCombiners:
    """Mean of several combiners' scores, standardised first so none of them dominates."""

    def __init__(self, factories):
        self.factories = factories

    def fit(self, X, y):
        self.fitted_ = [f().fit(X, y) for f in self.factories]
        s = [m.score(X) for m in self.fitted_]
        self.mu_ = [float(v.mean()) for v in s]
        self.sd_ = [float(max(v.std(), 1e-9)) for v in s]
        return self

    def score(self, X):
        return np.mean([(m.score(X) - mu) / sd
                        for m, mu, sd in zip(self.fitted_, self.mu_, self.sd_)], axis=0)


def main():
    members = pinned_members(HERE)
    tags = [m["tag"] for m in members]
    test_ids, X, y = aligned(members, "test")
    docs, _ = doc_structure(test_ids, "test")
    print(f"clean pool: {len(tags)} members | test {X.shape}\n")

    cands = [
        ("greedy r25", lambda: C.GreedyMultiset(rounds=25)),
        ("greedy r40", lambda: C.GreedyMultiset(rounds=40)),
        ("bagged m.50 s1", lambda: C.BaggedGreedy(n_bags=15, model_frac=0.50, rounds=15)),
        ("bagged m.35 s1", lambda: C.BaggedGreedy(n_bags=15, model_frac=0.35, rounds=15)),
        ("bagged m.25 s1", lambda: C.BaggedGreedy(n_bags=15, model_frac=0.25, rounds=15)),
        ("bagged m.35 x3 seeds", lambda: MultiSeedBagged(n_seeds=3, n_bags=15, model_frac=0.35, rounds=15)),
        ("blend greedy+bagged+nnls", lambda: BlendCombiners([
            lambda: C.GreedyMultiset(rounds=25),
            lambda: C.BaggedGreedy(n_bags=15, model_frac=0.35, rounds=15),
            lambda: C.NNLS()])),
    ]

    rows = []
    for name, f in cands:
        t0 = time.time()
        q, _ = C.cv_qwk_full(f, X, y, docs, n_folds=5)
        rows.append({"method": name, "cv_qwk": q * 100, "secs": round(time.time() - t0)})
        print(f"{name:28s} CV={q*100:7.3f}  ({time.time()-t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: -r["cv_qwk"])
    print(f"\nbest: {rows[0]['method']}  CV={rows[0]['cv_qwk']:.3f}  "
          f"-> implied blind {rows[0]['cv_qwk']-1.42:.2f}  (need > 85.30)")
    json.dump({"n_pool": len(tags), "results": rows},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
