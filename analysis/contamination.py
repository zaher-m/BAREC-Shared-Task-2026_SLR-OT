"""How inflated is an _ad member's score on the split it early-stopped on?

The _ad members early-stop on the public test split, which is also where selection and
calibration happen. Early stopping keeps the best epoch, so the kept checkpoint sits above the
member's typical level on that split by construction. The size of that lift is what we call the
contamination: best-epoch QWK minus mean-epoch QWK, read from each member's training log.

This is the producer for the "1.06 QWK, n=18" figure in docs/method.md and exp010's notes.
n=18 because those are the _ad encoder logs with per-epoch records that survive under
artifacts/logs/; the six earliest _ad members trained before per-member logs were kept, and the
LLM log records no per-epoch QWK.

    python analysis/contamination.py
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slra_ot.paths import LOGS  # noqa: E402

PAT = re.compile(r"dev_QWK\(naive\)=([0-9.]+)")


def per_member():
    rows = []
    for f in sorted(LOGS.glob("train_*ad*.log")):
        qs = [float(m) for m in PAT.findall(f.read_text())]
        if len(qs) < 2:
            continue
        tag = f.name[len("train_"):-len(".log")]
        rows.append((tag, len(qs), max(qs), float(np.mean(qs))))
    return rows


def main():
    rows = per_member()
    print(f"{'member':28s} {'epochs':>7s} {'best':>8s} {'mean':>8s} {'best-mean':>10s}")
    for tag, n, best, mean in rows:
        print(f"{tag:28s} {n:7d} {best:8.3f} {mean:8.3f} {best-mean:10.3f}")
    d = np.array([b - m for _, _, b, m in rows])
    print(f"\nmean inflation over n={len(rows)} members: {d.mean():.3f} QWK "
          f"(sd {d.std():.3f}, min {d.min():.3f}, max {d.max():.3f})")
    print("this is the 1.06 quoted in the docs")


if __name__ == "__main__":
    main()
