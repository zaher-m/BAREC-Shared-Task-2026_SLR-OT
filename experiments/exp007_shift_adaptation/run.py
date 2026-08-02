"""exp007: is the -1.4 drop from public test to blind partly a fixable score shift?

Every team loses about the same going from public test to blind: this system 86.40 -> 85.00 and
86.62 -> 85.2, thylinao 86.30 -> 84.80. The usual reading is that blind is just harder, but
exp003 left a clue: the blind scores are compressed relative to the calibration set (sd 2.738
vs 2.880, mean 10.309 vs 10.451). Cuts fitted on the wider distribution then sit too far out
on the narrower one, so the extreme levels get under-predicted, and QWK punishes that
quadratically.

Two ways to read the compression:
  (a) it is real, blind has fewer extreme sentences, and stretching it back would invent
      variance that is not there;
  (b) it is an artefact, the members are less confident off-distribution so their expected
      levels shrink toward the mean, and undoing that is free QWK.

This can be settled without touching blind. Dev and test are both held out for the 39
train-only members, and they are two different document samples, so fitting on one and
predicting the other is the same kind of shift with labels on both sides.

  affine   : rescale target scores to the source mean and sd
  spread   : rescale the spread only, keep the target's own mean
  quantile : map target scores onto the source's quantiles

Only worth applying to blind if it wins in both directions.
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

from slra_ot import combiners as C  # noqa: E402
from slra_ot.members import aligned, pinned_members  # noqa: E402
from slra_ot.metrics import full_report  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402


def align(target, source, how):
    t = np.asarray(target, float)
    s = np.asarray(source, float)
    if how == "none":
        return t
    if how == "affine":
        return (t - t.mean()) / max(t.std(), 1e-9) * s.std() + s.mean()
    if how == "spread":
        return (t - t.mean()) / max(t.std(), 1e-9) * s.std() + t.mean()
    if how == "quantile":
        order = np.argsort(t)
        out = np.empty_like(t)
        qs = (np.arange(len(t)) + 0.5) / len(t)
        out[order] = np.quantile(np.sort(s), qs)
        return out
    raise ValueError(how)


def direction(name, Xsrc, ysrc, Xtgt, ytgt, rows):
    g = C.GreedyMultiset(rounds=25).fit(Xsrc, ysrc)
    s_src, s_tgt = g.score(Xsrc), g.score(Xtgt)
    th = C.fit_thresholds(s_src, ysrc)
    print(f"\n{name}: source mean={s_src.mean():.3f} sd={s_src.std():.3f} | "
          f"target mean={s_tgt.mean():.3f} sd={s_tgt.std():.3f} "
          f"(sd ratio {s_tgt.std()/s_src.std():.3f})")
    base = None
    for how in ["none", "affine", "spread", "quantile"]:
        rep = full_report(ytgt, labels_from_thresholds(align(s_tgt, s_src, how), th))
        if base is None:
            base = rep["QWK"]
        rows.append({"direction": name, "align": how, "qwk": rep["QWK"],
                     "delta": rep["QWK"] - base, "acc19": rep["Acc19"], "mae": rep["MAE"]})
        print(f"  align={how:9s} QWK={rep['QWK']:7.3f} ({rep['QWK']-base:+6.3f})  "
              f"acc19={rep['Acc19']:5.2f} mae={rep['MAE']:.3f}", flush=True)


def main():
    p39 = [m for m in pinned_members(HERE) if not m["tag"].endswith(("_ad", "_ad2"))]
    print(f"{len(p39)} train-only members (dev and test both genuinely held out)")
    _, Xd, yd = aligned(p39, "dev")
    _, Xt, yt = aligned(p39, "test")

    rows = []
    direction("dev -> test", Xd, yd, Xt, yt, rows)
    direction("test -> dev", Xt, yt, Xd, yd, rows)

    print("\nmean delta by alignment (positive = alignment helps):")
    verdict = {}
    for how in ["affine", "spread", "quantile"]:
        d = [r["delta"] for r in rows if r["align"] == how]
        verdict[how] = float(np.mean(d))
        print(f"  {how:9s} {np.mean(d):+.3f}   (per direction: {[f'{x:+.3f}' for x in d]})")
    best = max(verdict, key=verdict.get)
    print(f"\n-> {'APPLY ' + best if verdict[best] > 0.05 else 'DO NOT ALIGN'} "
          f"(best mean delta {verdict[best]:+.3f})")

    json.dump({"results": rows, "mean_delta": verdict},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
