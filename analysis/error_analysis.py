"""Where the QWK loss actually comes from, and whether blind looks shifted.

Four things, on a saved ensemble:
  1. cost per confusion cell, i.e. which mistakes carry the loss (not which are commonest)
  2. recall and mean signed error per level, to see where the scale is compressed
  3. predicted vs true level distribution on dev, since that is QWK's denominator
  4. predicted blind distribution vs the train distribution, as a shift check

    python analysis/error_analysis.py --ensemble artifacts/ensembles/v1.json \
        --submission sub07_clean_k035_silver
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slra_ot.members import load_members, aligned          # noqa: E402
from slra_ot.metrics import _W, qwk                        # noqa: E402
from slra_ot.paths import ENSEMBLES, RAW_SPLITS, submission_dir  # noqa: E402
from slra_ot.thresholds import QWKThresholdOptimizer       # noqa: E402


def ensemble_scores(ens, split):
    members = load_members(tags=ens["members"])
    w = np.asarray(ens["weights"], float)
    w = w / w.sum()
    _, X, y = aligned(members, split)
    return X @ w, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default=str(ENSEMBLES / "ensemble.json"))
    ap.add_argument("--submission", default=None,
                    help="submission directory name, for the blind-marginal comparison")
    args = ap.parse_args()

    ens = json.loads(Path(args.ensemble).read_text())
    dev_s, dev_y = ensemble_scores(ens, "dev")
    test_s, test_y = ensemble_scores(ens, "test")
    opt = QWKThresholdOptimizer().fit(dev_s, dev_y)
    dev_p, test_p = opt.predict(dev_s), opt.predict(test_s)
    print(f"ensemble dev QWK={qwk(dev_y, dev_p)*100:.3f}  "
          f"test QWK={qwk(test_y, test_p)*100:.3f}\n")

    # --- 1. per-cell QWK cost -------------------------------------------------
    O = np.zeros((19, 19))
    for t, p in zip(dev_y - 1, dev_p - 1):
        O[t, p] += 1
    cost = _W * O
    tot = cost.sum()
    print("=== Top-12 costliest confusion cells on DEV (true->pred: count, dist, cost-share%) ===")
    cells = [(i + 1, j + 1, int(O[i, j]), abs(i - j), cost[i, j] / tot * 100)
             for i in range(19) for j in range(19) if i != j and O[i, j] > 0]
    for t, p, c, dist, sh in sorted(cells, key=lambda x: -x[4])[:12]:
        print(f"  true {t:2d} -> pred {p:2d} : n={c:4d}  |dist|={dist}  cost={sh:4.1f}%")

    # --- 2. per-level recall and signed error --------------------------------
    print("\n=== Per-true-level (DEV): n, recall, mean signed error (pred-true) ===")
    for t in range(1, 20):
        mask = dev_y == t
        if mask.sum() == 0:
            continue
        print(f"  L{t:2d}: n={mask.sum():4d} recall={(dev_p[mask]==t).mean()*100:5.1f}%  "
              f"mean(pred-true)={(dev_p[mask]-t).mean():+.2f}")

    # --- 3. marginal match ---------------------------------------------------
    print("\n=== DEV marginal: true vs predicted share per level ===")
    tt = np.bincount(dev_y, minlength=20)[1:] / len(dev_y) * 100
    pp = np.bincount(dev_p, minlength=20)[1:] / len(dev_p) * 100
    print("  L  true%  pred%")
    for l in range(19):
        print(f"  {l+1:2d} {tt[l]:5.1f} {pp[l]:6.1f}")

    # --- 4. blind marginal vs train marginal ---------------------------------
    if not args.submission:
        return
    rec = submission_dir(args.submission) / "prediction"
    bp = pd.read_csv(rec) if rec.exists() else pd.read_csv(
        io.BytesIO(zipfile.ZipFile(str(rec) + ".zip").read("prediction")))
    tr = pd.read_parquet(RAW_SPLITS["train"])
    train_counts = np.bincount(tr["Readability_Level_19"].astype(int), minlength=20)[1:]
    train_dist = train_counts / len(tr) * 100
    blind_dist = np.bincount(bp["Prediction"].astype(int), minlength=20)[1:] / len(bp) * 100
    print(f"\n=== BLIND ({args.submission}) vs TRAIN level distribution ===")
    print("  L  train%  devTrue%  BLINDpred%   (blind-train)")
    for l in range(19):
        print(f"  {l+1:2d} {train_dist[l]:6.1f} {tt[l]:8.1f} {blind_dist[l]:10.1f}   "
              f"{blind_dist[l]-train_dist[l]:+.1f}")
    print(f"\n  mean predicted blind level: {bp['Prediction'].mean():.2f}")
    print(f"  mean train level:           {tr['Readability_Level_19'].mean():.2f}")
    print(f"  mean dev level:             {dev_y.mean():.2f}")


if __name__ == "__main__":
    main()
