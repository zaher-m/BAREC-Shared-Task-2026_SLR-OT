"""Fit an ensemble on the public test split and write the blind submission zip.

This script makes no decisions. It takes a configuration that an experiment has already
justified by document-grouped CV, refits it on the whole calibration set, applies it to the
cached blind scores and writes the zip. The choosing happens in experiments/.

    python -m slra_ot.cli.build_submission --name sub07_clean_k035_silver \
        --combiner greedy --clean-only --thresholds prior --prior-k 0.35 --cv-estimate 86.558
"""
import argparse
import io
import json
import zipfile

import numpy as np
import pandas as pd

from slra_ot import combiners as C
from slra_ot.members import (aligned, blind_ids, blind_matrix, doc_structure, load_members,
                             pinned_pool)
from slra_ot.metrics import full_report, qwk
from slra_ot.paths import CANDIDATES, RAW_SPLITS, REGENERATED, SUBMITTED, submission_dir
from slra_ot.thresholds import labels_from_thresholds

# Measured drop from public test to blind. Not a constant: it gets worse the more weight
# the system puts on the contaminated (_ad) members, because they inflate the CV estimate
# without helping as much on blind.
#   sub07  clean only   CV 86.558 -> 85.5   offset -1.06
#   sub06  30% _ad      CV 86.663 -> 85.3   offset -1.36
#   sub02  ~30% _ad     CV 86.623 -> 85.2   offset -1.42
#   sub03  majority     CV 86.796 -> 84.9   offset -1.90
# Using one offset for every pool is what made sub06 look better than sub07 beforehand.
BLIND_SHIFT_CLEAN = -1.06
BLIND_SHIFT_CAPPED = -1.40

AD = ("_ad", "_ad2", "_ad3", "_ad4")


def build_combiner(kind, seed=0):
    if kind == "greedy":
        return C.GreedyMultiset(rounds=25)
    if kind == "bagged":
        return C.BaggedGreedy(n_bags=25, model_frac=0.5, rounds=15, seed=seed)
    if kind == "bagged_rows":
        return C.BaggedGreedy(n_bags=25, model_frac=0.5, rounds=15, row_frac=0.8, seed=seed)
    raise ValueError(kind)


def apply_transform(kind, Xfit, Xblind):
    """Return the transformed (fit, blind) matrices. z uses fit-set stats for both."""
    if kind == "raw":
        return Xfit, Xblind
    if kind == "z":
        mu, sd = Xfit.mean(0), np.maximum(Xfit.std(0), 1e-6)
        return (Xfit - mu) / sd, (Xblind - mu) / sd
    if kind == "rank":
        from scipy.stats import norm, rankdata
        f = lambda A: np.column_stack([norm.ppf(rankdata(A[:, j]) / (len(A) + 1.0))
                                       for j in range(A.shape[1])])
        return f(Xfit), f(Xblind)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--combiner", default="bagged", choices=["greedy", "bagged", "bagged_rows"])
    ap.add_argument("--transform", default="raw", choices=["raw", "z", "rank"])
    ap.add_argument("--thresholds", default="single", choices=["single", "bagged", "prior"])
    ap.add_argument("--prior-k", type=float, default=0.5,
                    help="shrinkage of the cuts toward the label-prior quantiles (exp009/exp010)")
    ap.add_argument("--clean-only", action="store_true",
                    help="drop the `_ad` family, whose test scores are inflated ~1.06 QWK "
                         "by early stopping on the calibration split")
    ap.add_argument("--ad-cap", type=float, default=None,
                    help="cap total weight on `_ad` members instead of dropping them")
    ap.add_argument("--seeds", type=int, default=3,
                    help="average the combiner's weights over this many bagging seeds")
    ap.add_argument("--cv-estimate", type=float, default=None)
    ap.add_argument("--pool", default=None,
                    help="file with one member tag per line, e.g. an experiment's pool.txt. "
                         "Default is every member that has cached scores, which is not the "
                         "same set the shipped submissions were built from")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing submission instead of comparing against it")
    args = ap.parse_args()

    members = (load_members(tags=pinned_pool(args.pool), drop_degenerate=False)
               if args.pool else load_members())
    if args.clean_only:
        members = [m for m in members if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    flagged = np.array([t.endswith(AD) for t in tags])
    print(f"{len(tags)} members ({flagged.sum()} early-stopped on the calibration split)")
    test_ids, Xt, yt = aligned(members, "test")
    docs, _ = doc_structure(test_ids, "test")
    bids = blind_ids()
    Xb = blind_matrix(tags)
    assert Xb.shape[1] == Xt.shape[1] == len(tags)
    print(f"pool {len(tags)} members | fit {Xt.shape} | blind {Xb.shape}")

    Xtf, Xbf = apply_transform(args.transform, Xt, Xb)

    if args.ad_cap is not None:
        n_seeds = 1
        combiner = C.CappedGreedy(flagged, cap=args.ad_cap, rounds=30).fit(Xtf, yt)
        W = combiner.weights_
        print(f"weight on early-stopped members: {combiner.flagged_weight_*100:.1f}% "
              f"(cap {args.ad_cap*100:.0f}%)")
    else:
        n_seeds = args.seeds if args.combiner != "greedy" else 1
        W = np.zeros(len(tags))
        for s in range(n_seeds):
            W += build_combiner(args.combiner, seed=s).fit(Xtf, yt).weights_
        W /= n_seeds

    s_fit, s_blind = Xtf @ W, Xbf @ W
    if args.thresholds == "bagged":
        th = C.bagged_thresholds(s_fit, yt, docs, n_bags=25)
    elif args.thresholds == "prior":
        th = C.prior_shrunk_thresholds(s_fit, yt, k=args.prior_k)
    else:
        th = C.fit_thresholds(s_fit, yt)
    print(f"threshold rule: {args.thresholds}"
          + (f" (k={args.prior_k})" if args.thresholds == "prior" else "")
          + f" | min gap between cuts {np.min(np.diff(np.sort(th))):.3f}"
          + f" | empty levels {19 - len(np.unique(labels_from_thresholds(s_blind, th)))}")
    pred = labels_from_thresholds(s_blind, th).astype(int)
    insample = full_report(yt, labels_from_thresholds(s_fit, th))

    nz = {tags[j]: round(float(w), 5) for j, w in enumerate(W) if w > 1e-6}
    print(f"\n{len(nz)} members with non-zero weight (top 12):")
    for t, w in sorted(nz.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {w*100:5.2f}%  {t}")
    print(f"\nin-sample TEST QWK {insample['QWK']:.3f} (optimistic)")
    shift = BLIND_SHIFT_CLEAN if args.clean_only else BLIND_SHIFT_CAPPED
    if args.cv_estimate:
        print(f"CV estimate {args.cv_estimate:.3f} + offset {shift:+.2f} "
              f"-> implied blind {args.cv_estimate + shift:.2f}")
    print(f"blind score mean={s_blind.mean():.3f} sd={s_blind.std():.3f} | "
          f"fit mean={s_fit.mean():.3f} sd={s_fit.std():.3f}")

    rec = submission_dir(args.name)
    d = rec if rec else REGENERATED / "submissions" / args.name
    if (d / "prediction.zip").exists() and not args.force:
        # What was uploaded is tracked in the repo. Rebuilding the same name is normally a
        # reproduction check, not a new submission, so compare and leave the file alone.
        old = pd.read_csv(io.BytesIO(zipfile.ZipFile(d / "prediction.zip").read("prediction")))
        same = (old["Prediction"].values == pred).mean()
        print(f"\n{args.name} already exists, not overwritten (--force to replace it).")
        print(f"this rebuild matches the uploaded prediction on {same*100:.2f}% of rows")
        return
    d.mkdir(parents=True, exist_ok=True)
    path = d / "prediction"
    pd.DataFrame({"Sentence ID": bids, "Prediction": pred}).to_csv(path, index=False)
    with zipfile.ZipFile(str(path) + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction")
    path.unlink()
    (d / "meta.json").write_text(json.dumps(
        {"name": args.name,
         # --combiner is ignored when --ad-cap is given: a cap means CappedGreedy
         "combiner": "capped_greedy" if args.ad_cap is not None else args.combiner,
         "ad_cap": args.ad_cap, "transform": args.transform,
         "threshold_strategy": args.thresholds,
         "prior_k": args.prior_k if args.thresholds == "prior" else None,
         "bagging_seeds": n_seeds,
         "n_pool": len(tags), "weights": nz, "thresholds": th.tolist(),
         "cv_estimate_qwk": args.cv_estimate,
         "blind_offset_used": shift,
         "implied_blind_qwk": (args.cv_estimate + shift) if args.cv_estimate else None,
         "insample_test_qwk": insample["QWK"], "fit_on": "BAREC public test (7286)"}, indent=2))

    # read the zip back and check it is exactly what the submission site expects
    z = zipfile.ZipFile(str(path) + ".zip")
    df = pd.read_csv(io.BytesIO(z.read("prediction")))
    blind_ref = pd.read_parquet(RAW_SPLITS["blind"])
    ok = (z.namelist() == ["prediction"]
          and list(df.columns) == ["Sentence ID", "Prediction"]
          and list(df["Sentence ID"]) == list(blind_ref["Sentence ID"])
          and df.Prediction.between(1, 19).all() and not df.Prediction.isna().any())
    print(f"\nwrote {d}/prediction.zip  ({len(df)} rows)  validation: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("submission failed validation")

    others = [p for root in (SUBMITTED, CANDIDATES) for p in sorted(root.iterdir()) if p.is_dir()]
    for prevdir in others:
        prev = prevdir.name
        p = prevdir / "prediction.zip"
        if prev == args.name or not p.exists():
            continue
        o = pd.read_csv(io.BytesIO(zipfile.ZipFile(p).read("prediction")))
        print(f"  vs {prev:26s} exact {100*(o.Prediction.values == pred).mean():5.1f}%  "
              f"QWK {qwk(o.Prediction.values, pred)*100:.2f}")


if __name__ == "__main__":
    main()
