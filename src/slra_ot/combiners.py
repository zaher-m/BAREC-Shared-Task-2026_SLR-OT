"""Ways of turning M member score columns into one score.

Every combiner has the same fit(X, y) / score(X) interface and returns a continuous level,
so the thresholding step afterwards is the same for all of them. That is what makes the
bake-off in exp002 a fair comparison.

The threshold rules are re-exported from slra_ot.thresholds because the CV helpers at the
bottom need them and a combiner is never scored without one.
"""
import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from slra_ot.metrics import fast_qwk
from slra_ot.thresholds import (FastQWKThresholds, QWKThresholdOptimizer, bagged_thresholds,
                                fit_thresholds, labels_from_thresholds,
                                prior_shrunk_thresholds, prior_thresholds)

__all__ = ["UniformAvg", "GreedyMultiset", "CappedGreedy", "BaggedGreedy", "TopKUniform",
           "NNLS", "RidgeStack", "HGBStack", "doc_folds", "cv_qwk", "cv_qwk_full",
           "bagged_thresholds", "fit_thresholds", "labels_from_thresholds",
           "prior_shrunk_thresholds", "prior_thresholds"]


# ------------------------------------------------------------------ combiners

class UniformAvg:
    """Plain mean of a fixed member subset. The baseline."""
    name = "uniform_avg"

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y):
        return self

    def score(self, X):
        return X[:, self.cols].mean(1) if self.cols is not None else X.mean(1)


class GreedyMultiset:
    """Caruana forward selection with replacement. A member's weight is how often it was
    picked.

    Each candidate is scored by how good the best cuts on it are, so the inner loop fits
    thresholds a few thousand times and has to be cheap. Three ways to do that:

      default          the exact optimiser, capped at 6 sweeps
      grid_objective   the coarse `coarse=(rounds, grid)` value grid. This is what the search
                       used until 2026-08-02, so exp001 to exp003 need it to reproduce, and it
                       is roughly 30x slower
      cv_objective     document-level CV per candidate instead of one in-sample fit. Slower
                       still, but the search stops chasing threshold noise
    """
    name = "greedy_multiset"

    def __init__(self, rounds=30, coarse=(3, 0.1), groups=None, cv_objective=False, n_folds=4,
                 grid_objective=False):
        self.rounds, self.coarse = rounds, coarse
        self.groups, self.cv_objective, self.n_folds = groups, cv_objective, n_folds
        self.grid_objective = grid_objective
        self.weights_ = None

    def _obj(self, s, y):
        if self.cv_objective:
            return cv_qwk(s, y, self.groups, self.n_folds, rounds=self.coarse[0],
                          grid=self.coarse[1])
        if self.grid_objective:
            return QWKThresholdOptimizer(rounds=self.coarse[0],
                                         grid=self.coarse[1]).fit(s, y).best_qwk_
        return FastQWKThresholds(rounds=6).fit(s, y).best_qwk_

    def fit(self, X, y):
        n, m = X.shape
        singles = [(self._obj(X[:, j], y), j) for j in range(m)]
        singles.sort(reverse=True)
        picks = [singles[0][1]]
        cur = X[:, picks[0]].copy()
        best, best_picks = self._obj(cur, y), list(picks)
        for _ in range(self.rounds):
            cand = [(self._obj((cur + X[:, j]) / (len(picks) + 1), y), j) for j in range(m)]
            q, j = max(cand)
            picks.append(j)
            cur = cur + X[:, j]
            if q > best + 1e-9:
                best, best_picks = q, list(picks)
        w = np.zeros(m)
        for j in best_picks:
            w[j] += 1
        self.weights_ = w / w.sum()
        self.best_obj_ = best
        return self

    def score(self, X):
        return X @ self.weights_


class CappedGreedy:
    """Greedy selection with a ceiling on the total weight the flagged members can get.

    The _ad members early-stopped on the calibration split, so their scores there are about
    1.06 QWK too high and greedy buys too many of them. They are still trained on 13% more
    data though, and the blind scores at the time said the answer was somewhere in between:
    0% weight scored 85.0 (sub01), ~30% scored 85.2 (sub02), unrestricted scored 84.9
    (sub03).

    So the cap is set from those blind scores, not from anything computed on the
    contaminated split, because nothing computed there can see the inflation. Later
    superseded: dropping the flagged members entirely beat every cap (exp017).
    """
    name = "capped_greedy"

    def __init__(self, flagged, cap=0.30, rounds=25):
        self.flagged = np.asarray(flagged, bool)
        self.cap, self.rounds = cap, rounds

    def _obj(self, s, y):
        return FastQWKThresholds(rounds=6).fit(s, y).best_qwk_

    def fit(self, X, y):
        n, m = X.shape
        counts = np.zeros(m)
        clean_idx = np.flatnonzero(~self.flagged)          # seed unflagged: cap holds at step 1
        best_j = max(clean_idx, key=lambda j: self._obj(X[:, j], y))
        counts[best_j] = 1
        cur = X[:, best_j].copy()
        best_obj, best_counts = self._obj(cur, y), counts.copy()

        for _ in range(self.rounds):
            total = counts.sum()
            cand = []
            for j in range(m):
                if self.flagged[j] and (counts[self.flagged].sum() + 1) / (total + 1) > self.cap:
                    continue
                cand.append((self._obj((cur + X[:, j]) / (total + 1), y), j))
            if not cand:
                break
            q, j = max(cand)
            counts[j] += 1
            cur = cur + X[:, j]
            if q > best_obj + 1e-9:
                best_obj, best_counts = q, counts.copy()
        self.weights_ = best_counts / best_counts.sum()
        self.flagged_weight_ = float(self.weights_[self.flagged].sum())
        return self

    def score(self, X):
        return X @ self.weights_


class BaggedGreedy:
    """Bagged ensemble selection (Caruana et al. 2004).

    Plain greedy picks whichever members happen to look best on the calibration rows. With
    67 correlated members and 7,286 rows that is about 0.75 QWK of overfit (exp003). Each
    bag here runs the same greedy over a random subset of the members, so no member gets in
    on one lucky fit, and averaging the bags gives a flatter weighting.

    row_frac < 1 also subsamples the rows per bag. More independent bags, noisier objective
    inside each one.
    """
    name = "bagged_greedy"

    def __init__(self, n_bags=20, model_frac=0.5, rounds=15, row_frac=1.0, seed=0):
        self.n_bags, self.model_frac = n_bags, model_frac
        self.rounds, self.row_frac, self.seed = rounds, row_frac, seed

    def fit(self, X, y):
        n, m = X.shape
        y = np.asarray(y, int)
        rng = np.random.default_rng(self.seed)
        W = np.zeros(m)
        k = max(2, int(round(self.model_frac * m)))
        for b in range(self.n_bags):
            cols = rng.choice(m, size=k, replace=False)
            if self.row_frac < 1.0:
                rows = rng.choice(n, size=max(50, int(self.row_frac * n)), replace=False)
                Xb, yb = X[np.ix_(rows, cols)], y[rows]
            else:
                Xb, yb = X[:, cols], y
            g = GreedyMultiset(rounds=self.rounds).fit(Xb, yb)
            W[cols] += g.weights_
        self.weights_ = W / W.sum()
        return self

    def score(self, X):
        return X @ self.weights_


class TopKUniform:
    """Mean of the k best members by solo calibrated QWK.

    Only one fitted parameter, so it is the hardest to overfit. Here as the floor that any
    fancier weighting has to beat.
    """
    name = "topk"

    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        q = [FastQWKThresholds(rounds=6).fit(X[:, j], y).best_qwk_ for j in range(X.shape[1])]
        self.cols_ = np.argsort(q)[::-1][:self.k]
        w = np.zeros(X.shape[1])
        w[self.cols_] = 1.0 / len(self.cols_)
        self.weights_ = w
        return self

    def score(self, X):
        return X @ self.weights_


class NNLS:
    """Non-negative least squares onto the label, with an intercept."""
    name = "nnls"

    def fit(self, X, y):
        A = np.c_[X, np.ones(len(X))]
        w, _ = nnls(A, np.asarray(y, float))
        self.w_ = w
        return self

    def score(self, X):
        return np.c_[X, np.ones(len(X))] @ self.w_


class RidgeStack:
    name = "ridge"

    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.m_ = Ridge(alpha=self.alpha).fit(X, np.asarray(y, float))
        return self

    def score(self, X):
        return self.m_.predict(X)


class HGBStack:
    """Gradient-boosted stacker. Can use non-linear member disagreement, if there is any."""
    name = "hgb"

    def __init__(self, **kw):
        self.kw = dict(max_iter=300, learning_rate=0.06, max_depth=4,
                       min_samples_leaf=40, l2_regularization=1.0, **kw)

    def fit(self, X, y):
        self.m_ = HistGradientBoostingRegressor(**self.kw).fit(X, np.asarray(y, float))
        return self

    def score(self, X):
        return self.m_.predict(X)


# ------------------------------------------------------------------- CV utils

def doc_folds(groups, n_folds, seed=0):
    """Fold assignment that keeps a document on one side of the split."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    f = dict(zip(uniq, rng.permutation(len(uniq)) % n_folds))
    return np.asarray([f[g] for g in groups])


def cv_qwk(scores, y, groups, n_folds=4, seed=0, rounds=4, grid=0.05):
    """Out-of-fold QWK where only the thresholds are refit per fold.

    Only valid for combiners with nothing to fit. Anything with weights needs cv_qwk_full.
    """
    y = np.asarray(y, int)
    fold = doc_folds(groups, n_folds, seed) if groups is not None else \
        np.random.default_rng(seed).integers(0, n_folds, len(y))
    pred = np.empty(len(y), int)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0:
            continue
        th = fit_thresholds(scores[tr], y[tr], rounds=rounds, grid=grid)
        pred[te] = labels_from_thresholds(scores[te], th)
    return fast_qwk(y - 1, pred - 1)


def cv_qwk_full(combiner_factory, X, y, groups, n_folds=4, seed=0, thresholds="single"):
    """CV where the combiner is refit inside each fold, not just the thresholds.

    thresholds="bagged" also refits the cuts by subsampling inside the fold, so the number
    covers the whole procedure that will be applied to blind. A callable (scores, y) -> cuts
    is also accepted, for the rules that take a parameter.
    """
    y = np.asarray(y, int)
    fold = doc_folds(groups, n_folds, seed)
    pred = np.empty(len(y), int)
    oof = np.empty(len(y), float)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        c = combiner_factory().fit(X[tr], y[tr])
        s_tr, s_te = c.score(X[tr]), c.score(X[te])
        if callable(thresholds):
            th = thresholds(s_tr, y[tr])
        elif thresholds == "bagged":
            th = bagged_thresholds(s_tr, y[tr], np.asarray(groups)[tr])
        else:
            th = fit_thresholds(s_tr, y[tr])
        pred[te] = labels_from_thresholds(s_te, th)
        oof[te] = s_te
    return fast_qwk(y - 1, pred - 1), oof
