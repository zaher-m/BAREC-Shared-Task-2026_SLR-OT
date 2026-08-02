"""Check the fast threshold optimiser against the slow reference one.

FastQWKThresholds updates QWK incrementally as a cut slides one position. That is the only
piece of arithmetic here that is easy to get wrong, so it gets checked against the obvious
implementation on synthetic ordinal data.

Runs under pytest, or on its own: python tests/test_thresholds.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slra_ot.metrics import fast_qwk, qwk                                       # noqa: E402
from slra_ot.thresholds import (FastQWKThresholds, QWKThresholdOptimizer,       # noqa: E402
                                labels_from_thresholds, prior_shrunk_thresholds)


def synthetic(n=6000, seed=0):
    """Labels plus a noisy score, roughly the shape of a real member's output."""
    rng = np.random.default_rng(seed)
    y = np.clip(rng.normal(10, 3.2, n).round(), 1, 19).astype(int)
    return y + rng.normal(0, 1.6, n), y


def test_fast_matches_reference():
    for trial in range(3):
        s, y = synthetic(seed=trial)

        t0 = time.time()
        ref = QWKThresholdOptimizer().fit(s, y)
        t_ref = time.time() - t0

        t0 = time.time()
        fast = FastQWKThresholds().fit(s, y)
        t_fast = time.time() - t0

        q_ref = fast_qwk(y - 1, labels_from_thresholds(s, ref.thresholds_) - 1)
        q_fast = fast_qwk(y - 1, fast.predict(s) - 1)
        print(f"trial {trial}: reference {q_ref*100:.4f} in {t_ref:.2f}s | "
              f"fast {q_fast*100:.4f} in {t_fast:.3f}s | speedup {t_ref/t_fast:.0f}x")

        # checking every cut position should never do worse than checking a 0.02 grid
        assert q_fast >= q_ref - 2e-3, (q_fast, q_ref)
        # the incrementally-maintained QWK should match one computed from scratch
        assert abs(fast.best_qwk_ - q_fast) < 1e-9, (fast.best_qwk_, q_fast)


def test_fast_qwk_matches_sklearn():
    s, y = synthetic(seed=7)
    pred = np.clip(np.rint(s), 1, 19).astype(int)
    assert abs(fast_qwk(y - 1, pred - 1) - qwk(y, pred)) < 1e-9


def test_thresholds_are_monotonic_and_fill_the_scale():
    s, y = synthetic(seed=3)
    for th in (FastQWKThresholds().fit(s, y).thresholds_,
               prior_shrunk_thresholds(s, y, k=0.35)):
        assert len(th) == 18
        assert np.all(np.diff(th) >= 0), "cut points must be monotonic"
        assert labels_from_thresholds(s, th).min() >= 1
        assert labels_from_thresholds(s, th).max() <= 19


def test_prior_shrinkage_widens_the_narrowest_gap():
    """This is why the shrunk rule transfers: cuts stop landing on top of each other."""
    s, y = synthetic(seed=11)
    free = np.sort(FastQWKThresholds().fit(s, y).thresholds_)
    shrunk = np.sort(prior_shrunk_thresholds(s, y, k=0.5))
    print(f"min gap between adjacent cuts: free {np.diff(free).min():.4f} "
          f"-> shrunk {np.diff(shrunk).min():.4f}")
    assert np.diff(shrunk).min() >= np.diff(free).min()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
