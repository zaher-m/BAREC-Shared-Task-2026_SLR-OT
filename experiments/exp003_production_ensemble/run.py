"""exp003: pick the ensemble to submit and write its blind prediction.

Two candidates, which cannot be compared on a single number:

  P39  39 members trained on the BAREC train split only. Dev and test are both held out, so
       weights can be fitted on dev and test reports cleanly.
  P45  the same plus the 6 _ad members, trained on train+dev (+13% data). Dev is inside
       their training set, so test is the only split held out for the whole pool.

To put them on the same footing, both go through document-grouped 5-fold CV on test with the
selection and the cuts refitted inside every fold (combiners.cv_qwk_full). That scores the
procedure rather than one lucky fit.

Writes the chosen pool's blind prediction into the submissions record.

Later note: the +0.180 this found for the _ad members was early-stopping inflation, not
skill. See exp010.
"""
import json
import os
import sys
import zipfile

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
from slra_ot.members import aligned, blind_ids, blind_matrix, doc_structure, pinned_members  # noqa: E402
from slra_ot.metrics import full_report, qwk  # noqa: E402
from slra_ot.paths import SUBMITTED, submission_dir  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

# This ran before FastQWKThresholds existed (2026-08-02), so both the free cuts and the
# greedy search used the older value grid. Asking for them explicitly is what keeps the numbers
# in results.json reproducible now that the exact optimiser is the default.
GRID = 0.02
CUTS = lambda s, y: C.fit_thresholds(s, y, rounds=8, grid=GRID)
GREEDY = lambda rounds: C.GreedyMultiset(rounds=rounds, grid_objective=True)


def write_submission(name, ids, pred, meta):
    """Write the blind prediction, or compare against the one already there.

    The uploaded zip is kept in the repo, so a re-run must not quietly replace it. If it
    exists this compares instead, which is the more useful outcome anyway: it says whether
    the experiment still produces what was submitted.
    """
    rec = submission_dir(name)
    d = str(rec) if rec else os.path.join(OUT_DIR, name)
    existing = os.path.join(d, "prediction.zip")
    if os.path.exists(existing):
        import io
        old = pd.read_csv(io.BytesIO(zipfile.ZipFile(existing).read("prediction")))
        same = (old["Prediction"].values == pred.astype(int)).mean()
        print(f"\n{name} already exists, not overwritten. "
              f"re-run matches the uploaded prediction on {same*100:.2f}% of rows")
        return d
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "prediction")
    pd.DataFrame({"Sentence ID": ids, "Prediction": pred.astype(int)}).to_csv(path, index=False)
    with zipfile.ZipFile(path + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction")
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    os.remove(path)
    print(f"\nwrote {d}/prediction.zip  ({len(ids)} rows)")
    return d


def main():
    all_members = pinned_members(HERE)
    p39 = [m for m in all_members if not m["tag"].endswith("_ad")]
    p45 = all_members
    tags39 = [m["tag"] for m in p39]
    tags45 = [m["tag"] for m in p45]

    dev_ids, Xd39, yd = aligned(p39, "dev")
    test_ids, Xt39, yt = aligned(p39, "test")
    _, Xt45, yt45 = aligned(p45, "test")
    assert (yt == yt45).all()
    test_docs, _ = doc_structure(test_ids, "test")
    print(f"P39 test {Xt39.shape} | P45 test {Xt45.shape} | {len(np.unique(test_docs))} test docs\n")

    # ---- same protocol for both pools --------------------------------------
    print("document-grouped 5-fold CV on TEST (greedy + thresholds refit per fold):")
    results = {}
    for label, X, tags in [("P39", Xt39, tags39), ("P45 (+_ad)", Xt45, tags45)]:
        q, _ = C.cv_qwk_full(lambda: GREEDY(25), X, yt, test_docs, n_folds=5,
                             thresholds=CUTS)
        results[label] = q * 100
        print(f"  {label:12s} CV QWK = {q*100:.3f}")
    delta = results["P45 (+_ad)"] - results["P39"]
    print(f"  -> the 6 train+dev members are worth {delta:+.3f} QWK\n")

    # ---- reference point: P39 fitted on dev, scored on test ----------------
    g = GREEDY(30).fit(Xd39, yd)
    th_dev = CUTS(g.score(Xd39), yd)
    rep = full_report(yt, labels_from_thresholds(g.score(Xt39), th_dev))
    print(f"reference  P39 fit-on-DEV -> TEST QWK = {rep['QWK']:.3f}  (exp002's best)\n")

    # ---- build the production system ---------------------------------------
    pool_tags, Xfit = (tags45, Xt45) if delta > 0 else (tags39, Xt39)
    pool_name = "P45" if delta > 0 else "P39"
    print(f"production pool: {pool_name} ({len(pool_tags)} members), weights+thresholds fit on TEST")
    gp = GREEDY(30).fit(Xfit, yt)
    s_fit = gp.score(Xfit)
    th = CUTS(s_fit, yt)
    insample = full_report(yt, labels_from_thresholds(s_fit, th))
    chosen = {pool_tags[j]: float(w) for j, w in enumerate(gp.weights_) if w > 0}
    print(f"  {len(chosen)} members with non-zero weight:")
    for t, w in sorted(chosen.items(), key=lambda kv: -kv[1]):
        print(f"    {w*100:5.1f}%  {t}")
    print(f"  in-sample TEST QWK = {insample['QWK']:.3f}  (optimistic; the CV number above is the estimate)")

    # ---- apply to blind ----------------------------------------------------
    bids, Xb = blind_ids(), blind_matrix(pool_tags)
    s_blind = Xb @ gp.weights_
    pred = labels_from_thresholds(s_blind, th).astype(int)
    print(f"\nblind score: mean={s_blind.mean():.3f} sd={s_blind.std():.3f} "
          f"(test fit: mean={s_fit.mean():.3f} sd={s_fit.std():.3f})")
    dist = pd.Series(pred).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(yt).value_counts(normalize=True).sort_index()
    print("predicted blind distribution vs TEST gold distribution:")
    for lv in range(1, 20):
        print(f"  {lv:2d}  blind {dist.get(lv,0)*100:5.2f}%   test-gold {test_dist.get(lv,0)*100:5.2f}%")

    meta = {
        "experiment": "exp003_production_ensemble",
        "pool": pool_name,
        "n_members_pool": len(pool_tags),
        "members": chosen,
        "thresholds": th.tolist(),
        "fit_on": "BAREC public TEST (7286 sentences)",
        "cv_estimate_qwk": results[pool_name if pool_name == "P39" else "P45 (+_ad)"],
        "insample_test_qwk": insample["QWK"],
        "reference_p39_fit_dev_test_qwk": rep["QWK"],
    }
    write_submission("sub02_pool45_greedy", bids, pred, meta)
    json.dump({"cv": results, "reference_p39_dev_fit": rep, "production": meta},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

    # sanity check: how much does this differ from the 85.00 submission?
    p01 = os.path.join(SUBMITTED, "sub01_st_v5_transfer", "prediction")
    if True:
        import io
        prev = pd.read_csv(p01) if os.path.exists(p01) else pd.read_csv(
            io.BytesIO(zipfile.ZipFile(p01 + ".zip").read("prediction")))
        import io
        prev = pd.read_csv(io.BytesIO(z.read("prediction")))
    same = (prev["Prediction"].values == pred).mean()
    print(f"\nagreement with sub01 (blind QWK 85.00): {same*100:.1f}% exact, "
          f"QWK between them {qwk(prev['Prediction'].values, pred)*100:.2f}")


if __name__ == "__main__":
    main()
