"""exp001: does document context add anything on top of the sentence ensemble?

Uses the frozen SLRA-ST v5 ensemble (the system that scored 85.00 on blind) and its cached
dev/test scores, so this tests the post-processing only and needs no GPU.

beta and the window size are chosen on dev with document-level 5-fold CV, so the 18 cuts are
never fitted and scored on the same rows. Test is reported once per scheme.

Worth checking the diagnostic first: the intra-document correlation of the residual
(y - score). If that is near zero, the encoders have already used whatever the document has
to offer and no amount of smoothing will help.
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

from slra_ot import doc_context  # noqa: E402
from slra_ot.members import aligned, doc_structure, load_members  # noqa: E402
from slra_ot.metrics import full_report, qwk  # noqa: E402
from slra_ot.thresholds import QWKThresholdOptimizer  # noqa: E402

# frozen SLRA-ST v5 ensemble (blind QWK 85.00)
V5_MEMBERS = ["arabertv2_corn_s2", "arabertv2_emd_d3", "arabertv2_large_corn",
              "arabertv2_large_reg", "arabertv2_large_soft", "arabertv2_soft_d3",
              "arabertv2_wkl_d3", "araelectra_corn_d3", "aramodern_reg_raw",
              "arbertv2_corn_d3"]
V5_WEIGHTS = np.array([1, 1, 1, 1, 2, 1, 1, 2, 2, 1], float)


def ensemble_score(members, split, weights):
    ids, X, y = aligned(members, split)
    return ids, X @ (weights / weights.sum()), y


def icc_residual(resid, doc_ids):
    """How much of the residual variance sits between documents rather than within."""
    resid = np.asarray(resid, float)
    grand = resid.mean()
    between = within = 0.0
    for sel in doc_context._group_slices(doc_ids):
        v = resid[sel]
        between += len(v) * (v.mean() - grand) ** 2
        within += ((v - v.mean()) ** 2).sum()
    return between / (between + within)


def dev_cv_qwk(score, y, docs, n_folds=5, seed=0):
    """CV on dev by document: fit the cuts on 4/5 of the documents, score the rest."""
    uniq = np.unique(docs)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.integers(0, n_folds, len(uniq))))
    fold = np.asarray([fold_of[d] for d in docs])
    preds = np.empty(len(y), int)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0:
            continue
        opt = QWKThresholdOptimizer(rounds=4, grid=0.05).fit(score[tr], y[tr])
        preds[te] = opt.predict(score[te])
    return qwk(y, preds)


def evaluate(name, dev_s, dev_y, dev_docs, test_s, test_y, results):
    dev_cv = dev_cv_qwk(dev_s, dev_y, dev_docs)
    opt = QWKThresholdOptimizer().fit(dev_s, dev_y)
    rep = full_report(test_y, opt.predict(test_s))
    results.append({"scheme": name, "dev_cv_qwk": dev_cv * 100,
                    "dev_fit_qwk": qwk(dev_y, opt.predict(dev_s)) * 100, **rep})
    print(f"{name:34s} devCV={dev_cv*100:6.3f}  TEST={rep['QWK']:6.3f} "
          f"(acc19={rep['Acc19']:.2f} adj={rep['Adj+-1']:.2f} mae={rep['MAE']:.3f})", flush=True)
    return dev_cv, rep["QWK"]


def main():
    members = load_members(tags=V5_MEMBERS)
    assert len(members) == len(V5_MEMBERS), [m["tag"] for m in members]
    dev_ids, dev_s, dev_y = ensemble_score(members, "dev", V5_WEIGHTS)
    test_ids, test_s, test_y = ensemble_score(members, "test", V5_WEIGHTS)
    dev_docs, dev_pos = doc_structure(dev_ids, "dev")
    test_docs, test_pos = doc_structure(test_ids, "test")
    print(f"dev  n={len(dev_y)} docs={len(np.unique(dev_docs))}")
    print(f"test n={len(test_y)} docs={len(np.unique(test_docs))}\n")

    # ---- the diagnostic that decides whether any of this can work -----------
    diag = {
        "dev_residual_icc": icc_residual(dev_y - dev_s, dev_docs),
        "test_residual_icc": icc_residual(test_y - test_s, test_docs),
        "dev_label_icc": icc_residual(dev_y.astype(float), dev_docs),
        "test_label_icc": icc_residual(test_y.astype(float), test_docs),
    }
    print("intra-document correlation")
    for k, v in diag.items():
        print(f"  {k:22s} {v:.4f}")
    print("  (label_icc = how clustered the labels are; residual_icc = how much of")
    print("   that clustering the sentence model has NOT already captured)\n")

    results = []
    evaluate("baseline (no doc context)", dev_s, dev_y, dev_docs, test_s, test_y, results)

    print("\n-- global document shrinkage --")
    for beta in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        evaluate(f"shrink beta={beta:.2f}",
                 doc_context.shrink(dev_s, dev_docs, beta), dev_y, dev_docs,
                 doc_context.shrink(test_s, test_docs, beta), test_y, results)

    print("\n-- empirical-Bayes shrinkage --")
    for cap in [0.15, 0.25, 0.35, 0.50]:
        evaluate(f"eb cap={cap:.2f}",
                 doc_context.eb_shrink(dev_s, dev_docs, cap=cap), dev_y, dev_docs,
                 doc_context.eb_shrink(test_s, test_docs, cap=cap), test_y, results)

    print("\n-- local neighbour smoothing --")
    for hw in [2, 5, 10, 20]:
        for beta in [0.10, 0.20, 0.30, 0.40]:
            evaluate(f"neighbor hw={hw:<2d} beta={beta:.2f}",
                     doc_context.neighbor_smooth(dev_s, dev_docs, dev_pos, beta, hw),
                     dev_y, dev_docs,
                     doc_context.neighbor_smooth(test_s, test_docs, test_pos, beta, hw),
                     test_y, results)

    best = max(results, key=lambda r: r["dev_cv_qwk"])
    print(f"\nbest by DEV-CV: {best['scheme']}  devCV={best['dev_cv_qwk']:.3f} "
          f"TEST={best['QWK']:.3f}  (baseline TEST={results[0]['QWK']:.3f}, "
          f"delta={best['QWK']-results[0]['QWK']:+.3f})")

    json.dump({"diagnostic": diag, "results": results, "best_by_dev_cv": best},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
    print(f"wrote {os.path.join(OUT_DIR, 'results.json')}")


if __name__ == "__main__":
    main()
