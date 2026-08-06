"""If the blind label distribution is shifted, which cut rule is safer?

Distribution matching only needs the continuous score: put the 18 cuts at the quantiles of
some target distribution T and the predicted distribution equals T by construction. The
target used here is a blend

    T = (1 - alpha) * p_dev + alpha * p_train

and alpha is picked by best worst-case QWK over simulated shifts of dev, since the domain
mix of the blind set is unknown.

    python analysis/label_shift_robustness.py --ensemble artifacts/ensembles/v1.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slra_ot.members import aligned, load_members             # noqa: E402
from slra_ot.metrics import fast_qwk                          # noqa: E402
from slra_ot.paths import ENSEMBLES, RAW_SPLITS               # noqa: E402
from slra_ot.thresholds import QWKThresholdOptimizer, labels_from_thresholds  # noqa: E402

rng = np.random.RandomState(0)


def distmatch_thresholds(scores, T):
    F = np.cumsum(T)[:-1]                      # 18 cumulative cut-points
    return np.quantile(scores, np.clip(F, 0, 1))


def qwk_of(scores, y, th):
    return fast_qwk(y - 1, labels_from_thresholds(scores, th) - 1)


def tilt(p, k):
    """Tilt a label distribution toward easy (k<0) or hard (k>0) levels."""
    lv = np.arange(1, 20)
    q = p * np.exp(k * (lv - 10) / 9.0)
    return q / q.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default=str(ENSEMBLES / "ensemble.json"))
    ap.add_argument("--n", type=int, default=6000, help="rows per simulated shifted sample")
    args = ap.parse_args()

    ens = json.loads(Path(args.ensemble).read_text())
    w = np.asarray(ens["weights"], float); w /= w.sum()
    _, Xd, dev_y = aligned(load_members(tags=ens["members"]), "dev")
    dev_s = Xd @ w

    tr = pd.read_parquet(RAW_SPLITS["train"])
    p_train = np.bincount(tr["Readability_Level_19"].astype(int), minlength=20)[1:].astype(float)
    p_train /= p_train.sum()
    p_dev = np.bincount(dev_y, minlength=20)[1:].astype(float)
    p_dev /= p_dev.sum()

    th_qwk = QWKThresholdOptimizer().fit(dev_s, dev_y).thresholds_
    print(f"unshifted dev: QWK-thresh={qwk_of(dev_s, dev_y, th_qwk)*100:.3f}  "
          f"distmatch(p_dev)={qwk_of(dev_s, dev_y, distmatch_thresholds(dev_s, p_dev))*100:.3f}  "
          f"distmatch(p_train)="
          f"{qwk_of(dev_s, dev_y, distmatch_thresholds(dev_s, p_train))*100:.3f}")

    shifts = {f"tilt{k:+.1f}": tilt(p_dev, k) for k in [-1.5, -0.75, 0, 0.75, 1.5]}
    idx_by = [np.where(dev_y == l)[0] for l in range(1, 20)]

    def resample_to(target):
        counts = (target * args.n).astype(int)
        picks = [rng.choice(idx_by[l], counts[l], replace=True)
                 for l in range(19) if len(idx_by[l]) and counts[l]]
        idx = np.concatenate(picks)
        return dev_s[idx], dev_y[idx]

    samples = [resample_to(t) for t in shifts.values()]

    print("\nmethod                         mean_QWK  worstcase_QWK  (across 5 simulated shifts)")
    res = [qwk_of(rs, ry, th_qwk) for rs, ry in samples]
    print(f"dev-QWK-thresholds (fixed)     {np.mean(res)*100:7.3f}   {np.min(res)*100:7.3f}")
    for a in [0, 0.25, 0.5, 0.75, 1.0]:
        T = (1 - a) * p_dev + a * p_train
        # cuts come from the shifted sample's own score quantiles, which is what would
        # happen at inference time on a distribution nothing was fitted on
        rr = [qwk_of(rs, ry, distmatch_thresholds(rs, T)) for rs, ry in samples]
        print(f"distmatch T=(1-a)dev+a*train a={a:.2f} "
              f"{np.mean(rr)*100:7.3f}   {np.min(rr)*100:7.3f}")


if __name__ == "__main__":
    main()
