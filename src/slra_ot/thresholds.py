"""Cutting a continuous score into levels 1..19.

Two optimisers (a slow reference one that searches a value grid, and a fast exact one)
plus four rules for where to put the 18 cuts.

The cuts are fitted on the same rows that picked the ensemble weights, so they overfit the
same way. The prior-shrunk rule below is what fixed that.
"""
import numpy as np

from slra_ot.metrics import N_CLASSES, fast_qwk

_W = (np.subtract.outer(np.arange(N_CLASSES), np.arange(N_CLASSES)) ** 2) / (N_CLASSES - 1) ** 2


def labels_from_thresholds(scores, thresholds):
    """Map continuous scores -> integer levels 1..19 via 18 monotonic cut points."""
    scores = np.asarray(scores, dtype=float)
    th = np.asarray(thresholds, dtype=float)
    return 1 + np.sum(scores[:, None] > th[None, :], axis=1)


class QWKThresholdOptimizer:
    """Coordinate ascent over the 18 cut points, maximising QWK on a held-out set.

    The top-ranked 2025 system reported +6.3 QWK from this kind of post-processing over naive
    rounding.

    The 0.02 grid turned out to help by accident: it is coarse enough to act as a
    regulariser, so this transfers better than the exact version below (exp009).
    """

    def __init__(self, n_classes=N_CLASSES, rounds=8, grid=0.02):
        self.n = n_classes
        self.rounds = rounds
        self.grid = grid
        self.thresholds_ = None

    def fit(self, scores, y_true):
        scores = np.asarray(scores, float)
        y0 = np.asarray([int(x) for x in y_true]) - 1  # 0-based for fast_qwk
        lo, hi = float(scores.min()) - 1.0, float(scores.max()) + 1.0
        # init at natural midpoints 1.5 .. 18.5, clipped to score range
        th = np.array([min(max(k + 0.5, lo), hi) for k in range(1, self.n)], float)
        best = fast_qwk(y0, labels_from_thresholds(scores, th) - 1)
        candidates = np.arange(lo, hi + self.grid, self.grid)
        for _ in range(self.rounds):
            improved = False
            for i in range(self.n - 1):
                cur = th[i]
                left = th[i - 1] if i > 0 else lo
                right = th[i + 1] if i < self.n - 2 else hi
                local = candidates[(candidates > left) & (candidates < right)]
                for c in local:
                    th[i] = c
                    score = fast_qwk(y0, labels_from_thresholds(scores, th) - 1)
                    if score > best + 1e-9:
                        best, cur, improved = score, c, True
                th[i] = cur
            if not improved:
                break
        self.thresholds_ = th
        self.best_qwk_ = best
        return self

    def predict(self, scores):
        return labels_from_thresholds(scores, self.thresholds_)


class FastQWKThresholds:
    """The same search, done exactly and much faster.

    The reference version re-scores every row for every candidate cut value: O(n) per
    candidate, ~10k candidates, about half a second per fit. Fine for one ensemble, too slow
    to bag the ensemble selection, which is where the 0.75 QWK of selection overfit from
    exp003 sits.

    The trick is to sort the scores once and treat the 18 cuts as indices into the sorted
    array. Moving a cut one position moves one sentence between two neighbouring levels, and
    both halves of QWK update in O(1):

        QWK = 1 - num/den,  num = sum_j W[y_j, pred_j],  den = sum_k Wt[k]*hp[k]/n
        Wt[k] = sum_j W[j,k]*ht[j]      (fixed once y is known)

    so moving a sentence from level a to level b costs
        num += W[y, b] - W[y, a]
        den += (Wt[b] - Wt[a]) / n

    One sweep over all 18 cuts is then O(n) instead of O(n * candidates), about 60-80x
    faster on 6k rows, and it checks every cut position rather than a grid. Note it is only
    better on the calibration set: it transfers worse (exp009), which is what
    prior_shrunk_thresholds is for.

    thresholds_ comes back as score values, so it can be applied to any other score vector
    with labels_from_thresholds. Checked against the reference in tests/test_thresholds.py.
    """

    def __init__(self, rounds=12, n_classes=N_CLASSES):
        self.rounds = rounds
        self.n = n_classes

    def fit(self, scores, y, init=None):
        scores = np.asarray(scores, float)
        y = np.asarray(y, int) - 1                        # 0..18
        n = len(scores)
        order = np.argsort(scores, kind="stable")
        s = scores[order]
        ys = y[order]

        ht = np.bincount(y, minlength=self.n).astype(float)
        Wt = _W.T @ ht                                    # Wt[k] = sum_j W[j,k]*ht[j]
        Wy = _W[ys]                                       # (n, 19) row per sentence

        # start from the label prior: cut where the cumulative gold distribution puts it
        c = ((np.cumsum(ht)[:-1] / n * n).astype(np.int64) if init is None
             else np.asarray(init, np.int64))
        c = np.clip(np.sort(c), 0, n)

        pred = np.searchsorted(c, np.arange(n), side="right")
        hp = np.bincount(pred, minlength=self.n).astype(float)
        num = float(Wy[np.arange(n), pred].sum())
        den = float(Wt @ hp / n)

        best_q = 1.0 - num / den if den > 0 else 0.0
        for _ in range(self.rounds):
            improved = False
            for i in range(self.n - 1):
                lo = c[i - 1] if i > 0 else 0
                hi = c[i + 1] if i < self.n - 2 else n
                start = int(c[i])
                best_pos, cur_best = start, best_q
                # sweep right: raising c[i] pulls sentence p from level i+1 down to i
                nm, dn = num, den
                for p in range(start, hi):
                    nm += Wy[p, i] - Wy[p, i + 1]
                    dn += (Wt[i] - Wt[i + 1]) / n
                    q = 1.0 - nm / dn if dn > 0 else 0.0
                    if q > cur_best + 1e-12:
                        cur_best, best_pos = q, p + 1
                # sweep left: lowering c[i] pushes sentence p-1 from level i up to i+1
                nm, dn = num, den
                for p in range(start, lo, -1):
                    nm += Wy[p - 1, i + 1] - Wy[p - 1, i]
                    dn += (Wt[i + 1] - Wt[i]) / n
                    q = 1.0 - nm / dn if dn > 0 else 0.0
                    if q > cur_best + 1e-12:
                        cur_best, best_pos = q, p - 1
                if best_pos != start:
                    # take the move, then rebuild both totals from scratch to avoid drift
                    c[i] = best_pos
                    pred = np.searchsorted(c, np.arange(n), side="right")
                    hp = np.bincount(pred, minlength=self.n).astype(float)
                    num = float(Wy[np.arange(n), pred].sum())
                    den = float(Wt @ hp / n)
                    best_q, improved = cur_best, True
            if not improved:
                break

        self.cuts_ = c
        self.best_qwk_ = best_q
        # turn cut indices into score values, midway between neighbouring scores
        pad = np.r_[s[0] - 1.0, s, s[-1] + 1.0]
        self.thresholds_ = (pad[c] + pad[c + 1]) / 2.0
        return self

    def predict(self, scores):
        scores = np.asarray(scores, float)
        return 1 + np.sum(scores[:, None] > self.thresholds_[None, :], axis=1)


# ---------------------------------------------------------------- threshold rules

def fit_thresholds(scores, y, rounds=12, grid=None):
    """Free fit, no shrinkage. Exact coordinate ascent over cut positions by default.

    Pass a grid spacing to get the slower value-grid optimiser instead. It is worth knowing
    which one a number came from: the grid version scores lower where it was fitted and
    transfers better (exp009), and it is what the experiments up to exp003 ran with, before
    FastQWKThresholds existed.
    """
    if grid is not None:
        return QWKThresholdOptimizer(rounds=rounds, grid=grid).fit(scores, y).thresholds_
    return FastQWKThresholds(rounds=rounds).fit(scores, y).thresholds_


def prior_thresholds(scores, y):
    """Distribution matching: cut at the label quantiles, ignoring QWK."""
    scores = np.asarray(scores, float)
    counts = np.bincount(np.asarray(y, int), minlength=20)[1:20].astype(float)
    cum = np.cumsum(counts / counts.sum())[:-1]
    return np.quantile(np.sort(scores), cum)


def prior_shrunk_thresholds(scores, y, k=0.5, rounds=12):
    """QWK-optimal cuts, pulled k of the way toward the cuts that match the label prior.

    18 cuts fitted freely on ~7k rows chase individual points. sub03 shipped with two cuts
    sitting on top of each other, predicted level 3 for 0.00% of the blind set and lost 5.7
    points of Acc19. Shrinking toward the prior quantiles means a cut only moves away from
    where the class distribution puts it if the data really pushes it.

    exp009 measured this on dev->test: +0.17 QWK and +2.2 Acc19 over a free fit, and the
    smallest gap between neighbouring cuts goes from 0.04 to 0.28, which is what stops a
    level from going empty.
    """
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    free = FastQWKThresholds(rounds=rounds).fit(scores, y).thresholds_
    counts = np.bincount(y, minlength=20)[1:20].astype(float)
    cum = np.cumsum(counts / counts.sum())[:-1]
    prior = np.quantile(np.sort(scores), cum)
    return np.sort((1.0 - k) * free + k * prior)


def bagged_thresholds(scores, y, groups=None, n_bags=15, frac=0.8, seed=0):
    """Mean of the cuts fitted on repeated subsamples of the calibration set.

    Cuts the data agrees on stay put, the rest get pulled toward the middle of their range.
    Subsampling is by document when groups are given, since sentences in the same document
    are not independent.
    """
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    rng = np.random.default_rng(seed)
    acc = []
    for b in range(n_bags):
        if groups is not None:
            uniq = np.unique(groups)
            keep = set(rng.choice(uniq, size=max(2, int(frac * len(uniq))), replace=False))
            rows = np.array([g in keep for g in groups])
        else:
            rows = rng.random(len(y)) < frac
        if rows.sum() < 100:
            continue
        acc.append(FastQWKThresholds(rounds=10).fit(scores[rows], y[rows]).thresholds_)
    return np.sort(np.mean(acc, axis=0))
