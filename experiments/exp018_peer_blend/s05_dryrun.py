"""exp018 step 5: test the blending machinery against gold before trusting it on blind.

The peer's quality on blind cannot be measured, there is no gold there. But three questions
about the mechanism can be measured on public test, and any of them could quietly eat the
gain:

  Q1  does blending in an extra predictor help at all when only ~53% of rows have it?
  Q2  does one global set of cuts survive the resulting heteroscedasticity? Covered rows get
      a less noisy score, so with shared cuts they get pulled toward the middle, and QWK
      cares about the predicted distribution.
  Q3  does the gain the variance model predicts, from measured D and v_o, match the gain that
      actually shows up? If theory over-predicts here, discount it on blind by the same
      factor.

Setup: take the strongest pool member out of the ensemble entirely, then hand it back on a
random 52.9% of test rows as a stand-in peer. The base ensemble is refitted without it, so any
gain is real and not greedy rediscovering a member it already had.

The stand-in shares training data with the pool even more than the real peer does, so the size
of the effect here is not the size on blind. What carries over is whether the machinery works
and whether the theory runs high or low.
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
from slra_ot.members import aligned, doc_structure, pinned_members  # noqa: E402
from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.thresholds import FastQWKThresholds, labels_from_thresholds  # noqa: E402

AD = ("_ad", "_ad2", "_ad3", "_ad4")
COVERAGE = 0.529


def blend(u, peer, have, w, expand=1.0):
    """Blend on the label scale, over the covered rows only.

    u and peer are both on the label scale. The peer's mean offset is removed on the covered
    rows, since a systematic shift is bias rather than information. expand pushes the blended
    rows back out from their own mean, to put back the variance the noise reduction took out:
    a less noisy score should be shrunk toward the centre less, and shared cuts applied to a
    compressed score under-predict the extremes.
    """
    out = u.copy()
    if not have.any() or w <= 0:
        return out
    p = peer[have] - peer[have].mean() + u[have].mean()
    b = (1 - w) * u[have] + w * p
    if expand != 1.0:
        b = b.mean() + (b - b.mean()) * expand
    out[have] = b
    return out


def theory_gain(D, v_o, c, w, coverage, base_qwk):
    rho = (1 + c - D / v_o) / (2 * np.sqrt(c))
    if rho > 1:
        return None, rho
    vr = (1 - w) ** 2 + c * w * w + 2 * w * (1 - w) * rho * np.sqrt(c)
    tot = coverage * vr + (1 - coverage)
    return 100 * (1 - (1 - base_qwk / 100) * tot), rho


def main():
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    ids, X, y = aligned(members, "test")
    docs, _ = doc_structure(ids, "test")

    # the strongest solo member becomes the stand-in peer and leaves the pool
    solo = [(FastQWKThresholds(rounds=6).fit(X[:, j], y).best_qwk_ * 100, j) for j in range(len(tags))]
    solo.sort(reverse=True)
    qh, jh = solo[0]
    print(f"synthetic peer = {tags[jh]} (solo test QWK {qh:.2f}), removed from the pool")
    keep = [j for j in range(len(tags)) if j != jh]
    Xk, peer_raw = X[:, keep], X[:, jh]

    rng = np.random.default_rng(0)
    have = rng.random(len(y)) < COVERAGE
    print(f"synthetic coverage {have.sum()}/{len(y)} ({100*have.mean():.1f}%)\n")

    # base ensemble, fit without the peer
    Wg = C.GreedyMultiset(rounds=30).fit(Xk, y).weights_
    s = Xk @ Wg
    a, b0 = np.polyfit(s, y, 1)
    u = a * s + b0                                    # base score on the label scale
    ap, bp = np.polyfit(peer_raw, y, 1)
    peer = ap * peer_raw + bp                         # peer on the label scale
    # cuts are fitted on the unblended score, same as they will be on blind
    th_raw = C.prior_shrunk_thresholds(s, y, k=0.35)
    # move the cuts onto the label scale so the same rule works on blended scores
    th = a * th_raw + b0 if a > 0 else np.sort(a * th_raw + b0)

    q0 = fast_qwk(y - 1, labels_from_thresholds(u, th) - 1) * 100
    print(f"base ensemble (47 members, peer excluded)  TEST QWK {q0:.3f}")

    v_o = float(np.var(y - u))
    D = float(np.mean((u[have] - (peer[have] - peer[have].mean() + u[have].mean())) ** 2))
    v_p = float(np.var(y - peer))
    c_true = v_p / v_o
    rho_true = float(np.corrcoef(y - u, y - peer)[0, 1])
    print(f"measurable:  v_o {v_o:.4f}   D {D:.4f}   D/v_o {D/v_o:.4f}")
    print(f"unmeasurable on blind, known here:  v_p {v_p:.4f}  c {c_true:.4f}  "
          f"rho(e_o,e_p) {rho_true:.4f}")
    est_rho = (1 + c_true - D / v_o) / (2 * np.sqrt(c_true))
    print(f"  rho recovered from D and c via the variance model: {est_rho:.4f} "
          f"(true {rho_true:.4f}) -> model error {est_rho-rho_true:+.4f}")

    print(f"\nQ1/Q2: realised TEST QWK by weight and expansion")
    print(f"{'w':>6s} " + " ".join(f"{'exp='+format(e,'.2f'):>10s}" for e in
                                   (1.00, 1.02, 1.04, 1.06)) + f" {'theory':>9s}")
    rows = []
    for w in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        line, best_e, best_q = [], None, -9
        for e in (1.00, 1.02, 1.04, 1.06):
            uu = blend(u, peer, have, w, expand=e)
            q = fast_qwk(y - 1, labels_from_thresholds(uu, th) - 1) * 100
            line.append(f"{q:10.3f}")
            if q > best_q:
                best_q, best_e = q, e
        tq, _ = theory_gain(D, v_o, c_true, w, have.mean(), q0)
        rows.append({"w": w, "realised_exp1": float(fast_qwk(
            y - 1, labels_from_thresholds(blend(u, peer, have, w), th) - 1) * 100),
            "best_expand": best_e, "best_qwk": float(best_q),
            "theory": float(tq) if tq else None})
        print(f"{w:6.2f} " + " ".join(line) + f" {tq:9.3f}")

    r1 = [r for r in rows if r["w"] > 0]
    real_best = max(r1, key=lambda r: r["realised_exp1"])
    th_best = max((r for r in r1 if r["theory"]), key=lambda r: r["theory"])
    print(f"\nQ3: calibration")
    print(f"  best realised: w={real_best['w']:.2f} -> {real_best['realised_exp1']:.3f} "
          f"({real_best['realised_exp1']-q0:+.3f} over base)")
    print(f"  theory at that w: {[r['theory'] for r in r1 if r['w']==real_best['w']][0]:.3f} "
          f"({[r['theory'] for r in r1 if r['w']==real_best['w']][0]-q0:+.3f})")
    gr = (real_best["realised_exp1"] - q0)
    gt = [r["theory"] for r in r1 if r["w"] == real_best["w"]][0] - q0
    ratio = gr / gt if gt else float("nan")
    print(f"  realised / theoretical gain = {ratio:.3f}")
    print(f"  theory optimum w={th_best['w']:.2f}; realised optimum w={real_best['w']:.2f}")
    exps = [r["best_expand"] for r in r1]
    print(f"  best expansion factor per w: {exps}  "
          f"-> {'expansion helps' if any(e>1.0 for e in exps) else 'expansion is a no-op'}")

    # document-grouped CV of the whole thing, in case the above is a test-split artefact
    print(f"\nCV check (document-grouped 5-fold, combiner+thresholds refit per fold)")
    fold = C.doc_folds(docs, 5, 0)
    for w in [0.0, 0.20, 0.30]:
        pred = np.empty(len(y), int)
        for f in range(5):
            tr, te = fold != f, fold == f
            Wf = C.GreedyMultiset(rounds=30).fit(Xk[tr], y[tr]).weights_
            sf_tr, sf_te = Xk[tr] @ Wf, Xk[te] @ Wf
            af, bf = np.polyfit(sf_tr, y[tr], 1)
            thf = af * C.prior_shrunk_thresholds(sf_tr, y[tr], k=0.35) + bf
            apf, bpf = np.polyfit(peer_raw[tr], y[tr], 1)
            uu = blend(af * sf_te + bf, apf * peer_raw[te] + bpf, have[te], w)
            pred[te] = labels_from_thresholds(uu, thf)
        print(f"  w={w:.2f}  CV QWK {fast_qwk(y-1, pred-1)*100:.3f}")

    json.dump({"peer_member": tags[jh], "peer_solo_qwk": qh, "coverage": float(have.mean()),
               "v_o": v_o, "D": D, "v_p": v_p, "c_true": c_true, "rho_true": rho_true,
               "rho_from_model": float(est_rho), "base_qwk": float(q0),
               "rows": rows, "realised_over_theory": float(ratio)},
              open(os.path.join(OUT_DIR, "s05_dryrun.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
