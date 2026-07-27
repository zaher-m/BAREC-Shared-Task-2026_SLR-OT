"""Pick members with greedy selection and save the result as an ensemble file.

Fits a combiner on one split, reports the other, and writes
artifacts/ensembles/<name>.json (members, weights, cuts) for slra_ot.cli.infer to use on new
sentences. Submissions built from the cached blind scores go through build_submission
instead.

    python -m slra_ot.cli.select_ensemble --fit-on dev --rounds 25 --name v1
"""
import argparse
import json

import numpy as np

from slra_ot import combiners as C
from slra_ot.members import aligned, doc_structure, load_members
from slra_ot.metrics import print_report
from slra_ot.paths import ENSEMBLES
from slra_ot.thresholds import FastQWKThresholds, labels_from_thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ensemble")
    ap.add_argument("--fit-on", default="dev", choices=["dev", "test"],
                    help="dev is only held out for members trained on the train split alone")
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--exclude", nargs="*", default=None, help="substrings of tags to exclude")
    ap.add_argument("--prior-k", type=float, default=0.35,
                    help="threshold shrinkage toward the label prior; None-like 0 = free fit")
    args = ap.parse_args()

    members = load_members(exclude=args.exclude)
    if args.fit_on == "dev":
        # the _ad members early-stopped on test, and their cached "dev" arrays ARE the test
        # split, so they cannot be fitted on dev at all; mixing them in mis-aligns the ids
        n0 = len(members)
        members = [m for m in members if not m["tag"].endswith(("_ad", "_ad2", "_ad3", "_ad4"))]
        if len(members) != n0:
            print(f"fit-on dev: dropped {n0 - len(members)} _ad members (dev is inside "
                  f"their training set)")
    tags = [m["tag"] for m in members]
    other = "test" if args.fit_on == "dev" else "dev"
    _, Xfit, yfit = aligned(members, args.fit_on)
    ids_other, Xother, yother = aligned(members, other)
    print(f"pool ({len(tags)} members), fitting on {args.fit_on} {Xfit.shape}\n")
    for j, t in enumerate(tags):
        print(f"  {t:28s} solo calibrated QWK="
              f"{FastQWKThresholds(rounds=6).fit(Xfit[:, j], yfit).best_qwk_*100:.3f}")

    g = C.GreedyMultiset(rounds=args.rounds).fit(Xfit, yfit)
    th = (C.prior_shrunk_thresholds(g.score(Xfit), yfit, k=args.prior_k) if args.prior_k
          else C.fit_thresholds(g.score(Xfit), yfit))
    chosen = {tags[j]: float(w) for j, w in enumerate(g.weights_) if w > 1e-9}
    print(f"\n{len(chosen)} members with non-zero weight:")
    for t, w in sorted(chosen.items(), key=lambda kv: -kv[1]):
        print(f"  {w*100:5.1f}%  {t}")

    print_report(f"{args.fit_on.upper()} (in-sample)", yfit,
                 labels_from_thresholds(g.score(Xfit), th))
    print_report(f"{other.upper()} (held out)", yother,
                 labels_from_thresholds(g.score(Xother), th))
    docs, _ = doc_structure(ids_other, other)
    q, _ = C.cv_qwk_full(lambda: C.GreedyMultiset(rounds=args.rounds), Xother, yother, docs,
                         n_folds=5)
    print(f"document-grouped 5-fold CV on {other.upper()}: {q*100:.3f}")

    ENSEMBLES.mkdir(parents=True, exist_ok=True)
    out = ENSEMBLES / f"{args.name}.json"
    out.write_text(json.dumps({
        "members": list(chosen), "weights": [chosen[t] for t in chosen],
        "weighting": "greedy-multiset", "fit_on": args.fit_on,
        "prior_k": args.prior_k, "thresholds": np.asarray(th).tolist(),
        f"cv_qwk_on_{other}": q * 100}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
