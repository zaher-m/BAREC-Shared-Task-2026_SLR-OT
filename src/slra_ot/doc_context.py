"""Document-level corrections applied to the ensemble score (exp001).

BAREC sentences come from documents, and the blind IDs give away the grouping even though
no metadata column ships with them. Sentences in one document share an author and an
audience, so a model that sees one sentence at a time is missing something that is free at
inference time. Worth a try, at least.

Three corrections, all applied to the continuous score before thresholding:

  shrink   : pull each score toward its document mean by a fixed factor
  eb       : same pull, empirical-Bayes weighted, so short documents (noisy means) move less
  neighbor : pull toward nearby sentences instead of the whole document, since register
             drifts over a long document

Each returns a new score vector; thresholds get refitted afterwards.

Result: none of them helped. Residual ICC is 0.07, so the encoders have already picked up
almost everything the document has to say.
"""
import numpy as np


def _group_slices(doc_ids):
    order = np.argsort(doc_ids, kind="stable")
    sorted_docs = np.asarray(doc_ids)[order]
    bounds = np.flatnonzero(np.r_[True, sorted_docs[1:] != sorted_docs[:-1], True])
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def doc_stats(scores, doc_ids):
    """Document-level stats broadcast back to each sentence."""
    scores = np.asarray(scores, float)
    n = len(scores)
    mean = np.empty(n); std = np.empty(n); size = np.empty(n)
    mx = np.empty(n); mn = np.empty(n); loo = np.empty(n)
    for sel in _group_slices(doc_ids):
        v = scores[sel]
        k = len(v)
        mean[sel] = v.mean()
        std[sel] = v.std() if k > 1 else 0.0
        size[sel] = k
        mx[sel] = v.max()
        mn[sel] = v.min()
        loo[sel] = (v.sum() - v) / (k - 1) if k > 1 else v
    return {"mean": mean, "std": std, "size": size, "max": mx, "min": mn, "loo": loo}


def shrink(scores, doc_ids, beta, use_loo=True):
    """s' = (1-beta)*s + beta*docmean(s).

    use_loo leaves the sentence out of its own document mean. Without it the correction is
    close to a no-op on short documents.
    """
    if beta == 0.0:
        return np.asarray(scores, float).copy()
    st = doc_stats(scores, doc_ids)
    target = st["loo"] if use_loo else st["mean"]
    return (1.0 - beta) * np.asarray(scores, float) + beta * target


def eb_shrink(scores, doc_ids, tau2=None, sigma2=None, cap=0.6, use_loo=True):
    """Shrink toward the document mean, weighted by how well that mean is estimated.

    w_d = tau2 / (tau2 + sigma2/n_d), so long documents get pulled harder than short ones.
    tau2 (between-document variance) and sigma2 (within-document) are estimated from the
    scores when not passed in. cap stops a very long document from collapsing onto its mean.
    """
    scores = np.asarray(scores, float)
    st = doc_stats(scores, doc_ids)
    if sigma2 is None:
        sigma2 = float(np.mean(st["std"] ** 2))
    if tau2 is None:
        uniq = {}
        for sel in _group_slices(doc_ids):
            uniq[len(uniq)] = scores[sel].mean()
        tau2 = max(float(np.var(list(uniq.values())) - sigma2 / np.mean(st["size"])), 1e-6)
    w = tau2 / (tau2 + sigma2 / np.maximum(st["size"], 1))
    w = np.minimum(w, cap)
    target = st["loo"] if use_loo else st["mean"]
    return (1.0 - w) * scores + w * target


def neighbor_smooth(scores, doc_ids, pos, beta, half_window=5, use_loo=True):
    """Pull toward the mean of the +-half_window sentences around it in the document."""
    if beta == 0.0:
        return np.asarray(scores, float).copy()
    scores = np.asarray(scores, float)
    out = scores.copy()
    for sel in _group_slices(doc_ids):
        idx = sel[np.argsort(np.asarray(pos)[sel])]
        v = scores[idx]
        k = len(v)
        cs = np.r_[0.0, np.cumsum(v)]
        lo = np.maximum(np.arange(k) - half_window, 0)
        hi = np.minimum(np.arange(k) + half_window + 1, k)
        tot = cs[hi] - cs[lo]
        cnt = (hi - lo).astype(float)
        if use_loo and k > 1:
            tot = tot - v
            cnt = np.maximum(cnt - 1.0, 1.0)
        out[idx] = (1.0 - beta) * v + beta * (tot / cnt)
    return out


def context_features(scores, doc_ids, pos, wc=None):
    """Features for a learned version of the above. Never got past the diagnostics."""
    scores = np.asarray(scores, float)
    st = doc_stats(scores, doc_ids)
    rel_pos = np.zeros_like(scores)
    for sel in _group_slices(doc_ids):
        k = len(sel)
        rel_pos[sel] = (np.asarray(pos)[sel] - np.asarray(pos)[sel].min()) / max(k - 1, 1)
    feats = [scores, st["loo"], scores - st["loo"], st["std"],
             np.log1p(st["size"]), st["max"], st["min"], rel_pos]
    names = ["score", "doc_loo", "resid", "doc_std", "log_docsize",
             "doc_max", "doc_min", "rel_pos"]
    if wc is not None:
        wc = np.asarray(wc, float)
        wcs = doc_stats(wc, doc_ids)
        feats += [np.log1p(wc), np.log1p(wcs["mean"])]
        names += ["log_wc", "log_doc_wc"]
    return np.vstack(feats).T, names
