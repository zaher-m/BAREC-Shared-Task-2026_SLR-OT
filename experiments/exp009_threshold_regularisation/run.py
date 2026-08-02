"""exp009: why sub03 lost, and which threshold rule transfers.

sub03 was submitted on a CV estimate of 86.796 (implied blind 85.38) and scored 84.9, below
sub02's 85.2. The pattern is clear: Acc19 fell 38.9 -> 33.2 while +-1 accuracy rose
72.8 -> 73.1. The scores did not get worse, the cuts moved. Comparing the two files, sub03
predicts level 3 for 0.00% of blind (sub02: 2.24%) and puts most of level 7 into level 6. Two
neighbouring cuts had collapsed onto each other.

Two things went wrong:

1. The exact threshold optimiser overfits. Parameterising cuts by position in the sorted
   array lets a cut land anywhere between two neighbouring scores, so 18 cuts can chase
   individual calibration points. The old 0.02 grid was coarse enough to stop that by
   accident. "Finds better optima" was true, and beside the point.

2. Within-test CV was the wrong tool. It resamples documents from the same split, so it
   measures variance inside one document sample, not decay onto a different one. It rated
   sub03 above sub02.

dev -> test is the better tool: two different document samples with labels on both sides, the
same shape as calibrate-on-test then predict-blind. It only works for the 39 train-only
members, which is fine, since the question here is the rule and not the pool.

The rules compared, all fitted on dev and applied to test:
  ref-grid    coordinate ascent on a 0.02 value grid (what sub02 used)
  exact       every cut position (what sub03 used)
  bagged      mean of exact fits over document subsamples
  prior-k     exact, then shrunk k of the way toward the cuts that reproduce the label prior
  minwidth    exact, but not allowed to collapse two cuts into an empty level
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
from slra_ot.metrics import full_report  # noqa: E402
from slra_ot.thresholds import FastQWKThresholds, labels_from_thresholds, QWKThresholdOptimizer  # noqa: E402


def prior_cuts(scores, y):
    """Cuts that make the predicted level distribution match the label distribution."""
    counts = np.bincount(np.asarray(y, int), minlength=20)[1:20].astype(float)
    cum = np.cumsum(counts / counts.sum())[:-1]
    return np.quantile(np.sort(np.asarray(scores, float)), cum)


def min_width_cuts(scores, y, min_frac=0.004):
    """Exact fit, then push apart any two cuts holding fewer than min_frac of the rows."""
    th = FastQWKThresholds(rounds=12).fit(scores, y).thresholds_.copy()
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    gap = max(int(min_frac * n), 1)
    idx = np.searchsorted(s, th)
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] < gap:
            idx[i] = min(idx[i - 1] + gap, n)
    pad = np.r_[s[0] - 1.0, s, s[-1] + 1.0]
    return (pad[idx] + pad[np.minimum(idx + 1, n + 1)]) / 2.0


def empty_levels(pred):
    return 19 - len(np.unique(pred))


def main():
    p39 = [m for m in pinned_members(HERE) if not m["tag"].endswith(("_ad", "_ad2", "_ad3"))]
    print(f"{len(p39)} train-only members (dev and test both genuinely held out)")
    dev_ids, Xd, yd = aligned(p39, "dev")
    test_ids, Xt, yt = aligned(p39, "test")
    dev_docs, _ = doc_structure(dev_ids, "dev")

    rows = []
    for cname, factory in [("greedy", lambda: C.GreedyMultiset(rounds=25)),
                           ("bagged greedy", lambda: C.BaggedGreedy(n_bags=20, model_frac=0.5, rounds=15))]:
        g = factory().fit(Xd, yd)
        sd_, st_ = g.score(Xd), g.score(Xt)

        rules = {
            "ref-grid": QWKThresholdOptimizer(rounds=8, grid=0.02).fit(sd_, yd).thresholds_,
            "exact": FastQWKThresholds(rounds=12).fit(sd_, yd).thresholds_,
            "bagged": C.bagged_thresholds(sd_, yd, dev_docs, n_bags=25, frac=0.8),
            "minwidth": min_width_cuts(sd_, yd),
        }
        pc = prior_cuts(sd_, yd)
        ex = rules["exact"]
        for k in (0.25, 0.50, 0.75):
            rules[f"prior-{k:.2f}"] = np.sort((1 - k) * ex + k * pc)
        rules["prior-1.00"] = np.sort(pc)

        print(f"\n--- combiner: {cname} ---")
        print(f"{'rule':12s} {'devQWK':>7s} {'TESTQWK':>8s} {'acc19':>6s} {'+-1':>6s} "
              f"{'MAE':>5s} {'emptyLv':>7s} {'minGap':>7s}")
        for rname, th in rules.items():
            th = np.sort(np.asarray(th, float))
            pt = labels_from_thresholds(st_, th)
            rep = full_report(yt, pt)
            devrep = full_report(yd, labels_from_thresholds(sd_, th))
            gap = float(np.min(np.diff(th)))
            rows.append({"combiner": cname, "rule": rname, "dev_qwk": devrep["QWK"],
                         "test_qwk": rep["QWK"], "acc19": rep["Acc19"], "adj": rep["Adj+-1"],
                         "mae": rep["MAE"], "empty_levels": empty_levels(pt), "min_gap": gap})
            print(f"{rname:12s} {devrep['QWK']:7.3f} {rep['QWK']:8.3f} {rep['Acc19']:6.2f} "
                  f"{rep['Adj+-1']:6.2f} {rep['MAE']:5.3f} {empty_levels(pt):7d} {gap:7.3f}",
                  flush=True)

    best = max(rows, key=lambda r: r["test_qwk"])
    print(f"\nbest dev->test transfer: {best['combiner']} + {best['rule']}  "
          f"TEST={best['test_qwk']:.3f} acc19={best['acc19']:.2f}")
    ref = next(r for r in rows if r["combiner"] == "greedy" and r["rule"] == "ref-grid")
    exa = next(r for r in rows if r["combiner"] == "greedy" and r["rule"] == "exact")
    print(f"sub02's rule (greedy+ref-grid) {ref['test_qwk']:.3f} | "
          f"sub03's rule (greedy+exact) {exa['test_qwk']:.3f} | "
          f"delta {exa['test_qwk']-ref['test_qwk']:+.3f}")
    json.dump({"results": rows}, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
