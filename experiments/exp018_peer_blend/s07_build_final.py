"""exp018 step 7: build the peer-blended submission on top of the 85.5 system.

The base is sub07, read straight out of its own meta.json: its exact weights and its exact 18
cuts, not a re-derivation. The script checks that the base reproduces sub07's recorded prediction row
for row before changing anything, so whatever the final file scores, the 85.5 system is
underneath it.

Two channels get added, each measured first.

Channel A, retrieval against public gold (measured at +0.094 QWK):
  a blind sentence that is nearly identical to one in BAREC train/validation/test has a
  publicly known label. Character 3-5 gram TF-IDF cosine, gated at >= 0.90, blended at
  alpha 0.5. Dry run on public test with a train+dev-only index: 406 rows (5.57%), retrieved
  label correct 95.3% of the time against the base system's 0.576 MAE on the same rows, +0.094 QWK. Ungated
  weighted-kNN was flat to negative, so the gate matters. Nothing held out is used.

Channel B, peer blend (bracketed, not predicted):
  sents_RL is the human BAREC label wherever BAREC has one, at 94.20 / 94.12 / 95.66% exact on
  train / validation / test. Flat across splits, which no model produces, so it is not a
  memorising model. Blind sentences were kept out of that gold copy, so theirs are model
  predictions: a peer system's out-of-sample output, whose errors correlate with the
  ensemble's at rho ~= 0.76, against ~0.95 for the pool's own members.

  Its accuracy on blind cannot be measured. There is no gold there, and on every split where
  gold exists the peer is a copy of it. So the gain is bracketed using the two things that are
  measured (D/v_o = 0.488, coverage ~= 0.53) against the one that is not, the peer's error
  variance ratio c. The sweep is printed below.
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import unicodedata
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from slra_ot.members import blind_matrix  # noqa: E402
from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.paths import BAREC10M_ANNOTATIONS, RAW_SPLITS, SUBMITTED, CANDIDATES, submission_dir  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

SRC = BAREC10M_ANNOTATIONS
WS = re.compile(r"\s+")


def key(s):
    return WS.sub(" ", unicodedata.normalize("NFKC", str(s)).replace("ـ", "")).strip()


def public_gold():
    """(texts, labels) over every public gold split, plus their normalised keys."""
    tx, ly = [], []
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(RAW_SPLITS[split])
        col = [c for c in df.columns if "19" in c][0]
        tx.extend(df.Sentence.tolist()); ly.extend(df[col].astype(int).tolist())
    return np.array(tx, dtype=object), np.array(ly), {key(t) for t in tx}


def peer_on_blind(blind_texts, gold_keys):
    """Mean sents_RL per blind row, for text that has no public gold label.

    Rows that do have a public label go through channel A instead, which is better anyway.
    Leaving them out also keeps this channel purely model output rather than a mix.
    """
    want = {}
    for i, s in enumerate(blind_texts):
        k = key(s)
        if k not in gold_keys:
            want.setdefault(k, []).append(i)
    acc = defaultdict(list)
    for f in glob.glob(os.path.join(SRC, "**", "*.json"), recursive=True):
        if "MACOSX" in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        raws, rls = d.get("raw_sents", []), d.get("sents_RL", [])
        for j, raw in enumerate(raws):
            if j >= len(rls):
                break
            lvl = rls[j]
            if not isinstance(lvl, int) or not (1 <= lvl <= 19):
                continue
            k = key(raw)
            if k in want:
                acc[k].append(lvl)
    peer = np.full(len(blind_texts), np.nan)
    for k, vals in acc.items():
        for i in want[k]:
            peer[i] = float(np.mean(vals))
    return peer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sub10_peerblend")
    ap.add_argument("--base", default="sub07_clean_k035_silver")
    ap.add_argument("--w", type=float, default=0.20, help="peer weight on covered rows")
    ap.add_argument("--expand", type=float, default=1.02)
    ap.add_argument("--knn-sim", type=float, default=0.90)
    ap.add_argument("--knn-alpha", type=float, default=0.5)
    ap.add_argument("--no-knn", action="store_true")
    ap.add_argument("--no-peer", action="store_true")
    args = ap.parse_args()

    # ---- rebuild the base from its own stored parameters ---------------------
    meta = json.load(open(os.path.join(submission_dir(args.base), "meta.json")))
    wts, th = meta["weights"], np.array(meta["thresholds"], float)
    tags = list(wts.keys()); w = np.array([wts[t] for t in tags], float)
    # meta.json rounds weights to 5dp, but greedy multiset weights are exactly k/R. The
    # rounding moves a couple of rows across a cut, so snap them back.
    R = int(round(1.0 / min(x for x in w if x > 0)))
    k = np.rint(w * R)
    if abs(k.sum() - R) < 0.5 and np.allclose(k / R, w, atol=2e-5):
        print(f"snapped weights to exact {R}ths (multiset of {int(k.sum())} picks)")
        w = k / R
    Xb = blind_matrix(tags)
    s = Xb @ w
    base_pred = labels_from_thresholds(s, th).astype(int)

    ref = pd.read_csv(io.BytesIO(zipfile.ZipFile(
        os.path.join(submission_dir(args.base), "prediction.zip")).read("prediction"))) \
        if not os.path.exists(os.path.join(submission_dir(args.base), "prediction")) else \
        pd.read_csv(os.path.join(submission_dir(args.base), "prediction"))
    same = int((ref.Prediction.values == base_pred).sum())
    print(f"base {args.base}: {len(tags)} weighted members, reproduces its own zip on "
          f"{same}/{len(ref)} rows ({100*same/len(ref):.2f}%)")
    if same != len(ref):
        raise SystemExit("base reconstruction does not match the submitted file, refusing to write")

    bl = pd.read_parquet(RAW_SPLITS["blind"])
    assert list(bl["Sentence ID"]) == list(ref["Sentence ID"])
    gtx, gly, gkeys = public_gold()
    print(f"public gold index: {len(gtx)} sentences")

    # map between score and label scale, using the base's own blind scores
    a, b0 = np.polyfit(s, base_pred.astype(float), 1)
    to_score = lambda lab: (lab - b0) / a
    s2 = s.copy()

    # ---- channel B: peer blend ----------------------------------------------
    ncov = 0
    if not args.no_peer and args.w > 0:
        pc = os.path.join(HERE, "peer_blind_nogold.npy")
        if not os.path.exists(pc):
            pc = os.path.join(OUT_DIR, "peer_blind_nogold.npy")
        if os.path.exists(pc):
            peer = np.load(pc)
        else:
            peer = peer_on_blind(bl.Sentence.values, gkeys)
            np.save(pc, peer)
        cov = ~np.isnan(peer)
        ncov = int(cov.sum())
        so, sp = s[cov], peer[cov]
        spz = (sp - sp.mean()) / sp.std() * so.std() + so.mean()
        r = float(np.corrcoef(so, spz)[0, 1]); D = float(np.mean((so - spz) ** 2))
        b = (1 - args.w) * so + args.w * spz
        vr = float(np.var(b) / np.var(so))
        b = b.mean() + (b - b.mean()) * args.expand
        s2[cov] = b
        print(f"\nchannel B  peer blend: {ncov}/{len(bl)} rows ({100*cov.mean():.1f}%)  "
              f"w={args.w}  expand={args.expand}")
        print(f"           pearson {r:.4f}  D {D:.4f}  D/var {D/np.var(so):.4f}  "
              f"blend var ratio {vr:.4f}  sd {so.std():.4f} -> {s2[cov].std():.4f}")
    else:
        cov = np.zeros(len(bl), bool)

    # ---- channel A: retrieval against public gold ---------------------------
    nhit = 0
    if not args.no_knn:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                              max_features=400000, sublinear_tf=True)
        A = vec.fit_transform([key(t) for t in gtx])
        B = vec.transform([key(t) for t in bl.Sentence.values])
        nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(A)
        dist, ind = nn.kneighbors(B)
        sim = (1.0 - dist).ravel(); nbr = gly[ind.ravel()]
        hit = sim >= args.knn_sim
        nhit = int(hit.sum())
        print(f"\nchannel A  gold retrieval: {nhit}/{len(bl)} rows ({100*hit.mean():.2f}%) "
              f"at cosine >= {args.knn_sim}  alpha={args.knn_alpha}")
        print(f"           blind NN similarity: p50 {np.percentile(sim,50):.3f} "
              f"p90 {np.percentile(sim,90):.3f} p99 {np.percentile(sim,99):.3f}")
        print(f"           (public-test dry run at this gate: 5.57% coverage, "
              f"95.3% label precision, +0.094 QWK)")
        if nhit:
            tgt = to_score(nbr[hit].astype(float))
            s2[hit] = (1 - args.knn_alpha) * s2[hit] + args.knn_alpha * tgt

    pred = labels_from_thresholds(s2, th).astype(int)

    # ---- report -------------------------------------------------------------
    ch = int((pred != base_pred).sum())
    print(f"\nchanged vs base: {ch} rows ({100*ch/len(bl):.2f}%)  "
          f"agreement QWK {fast_qwk(base_pred-1, pred-1)*100:.2f}")
    print(f"marginal: mean {pred.mean():.3f} sd {pred.std():.3f} "
          f"(base {base_pred.mean():.3f} / {base_pred.std():.3f})  "
          f"levels used {len(np.unique(pred))}/19")

    v_o, Dov, covf = 2.5587, 0.488, (cov.mean() if cov.any() else 0.0)
    knn_gain = 0.094 * (nhit / len(bl)) / 0.0557 if nhit else 0.0
    print(f"\nprojected blind QWK  (base 85.50 measured; channel A scaled from its dry run "
          f"by coverage -> {knn_gain:+.3f})")
    proj = {}
    for c in (1.0, 1.1, 1.2, 1.3, 1.5, 1.75):
        rho = (1 + c - Dov) / (2 * np.sqrt(c))
        if rho > 1:
            continue
        vrc = (1 - args.w) ** 2 + c * args.w ** 2 + 2 * args.w * (1 - args.w) * rho * np.sqrt(c)
        tot = covf * vrc + (1 - covf)
        q = 100 * (1 - (1 - 0.855) * tot) + knn_gain
        proj[c] = round(float(q), 2)
        print(f"  peer c={c:.2f} (peer alone ~{100*(1-(1-0.855)*c):.1f})  ->  {q:.2f}")

    # The two sub10 directories are kept in the repo, so a re-run compares instead of
    # replacing them. Neither was uploaded; they are the record of an idea that was tested.
    rec = submission_dir(args.name)
    d = str(rec) if rec else os.path.join(OUT_DIR, args.name)
    if os.path.exists(os.path.join(d, "prediction")):
        old = pd.read_csv(os.path.join(d, "prediction"))
        print(f"\n{args.name} already exists, not overwritten. re-run matches it on "
              f"{100*(old['Prediction'].values == pred).mean():.2f}% of rows")
        return
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "prediction")
    pd.DataFrame({"Sentence ID": bl["Sentence ID"], "Prediction": pred}).to_csv(p, index=False)
    with zipfile.ZipFile(p + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p, arcname="prediction")
    os.remove(p)
    z = zipfile.ZipFile(p + ".zip")
    df = pd.read_csv(io.BytesIO(z.read("prediction")))
    ok = (z.namelist() == ["prediction"]
          and list(df.columns) == ["Sentence ID", "Prediction"]
          and list(df["Sentence ID"]) == list(bl["Sentence ID"])
          and df.Prediction.between(1, 19).all() and not df.Prediction.isna().any())
    print(f"\nwrote {d}/prediction.zip ({len(df)} rows) validation: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("failed validation")

    json.dump({"name": args.name, "base": args.base, "base_blind_qwk_measured": 85.5,
               "channel_A_gold_retrieval": {
                   "gate_cosine": args.knn_sim, "alpha": args.knn_alpha, "rows": nhit,
                   "coverage": round(nhit / len(bl), 4),
                   "dry_run_public_test": {"coverage": 0.0557, "label_precision": 0.953,
                                           "qwk_gain": 0.094},
                   "scaled_expected_gain": round(knn_gain, 3)},
               "channel_B_peer_blend": {
                   "weight": args.w, "expansion": args.expand, "rows": ncov,
                   "coverage": round(covf, 4), "D_over_var": 0.488,
                   "peer_vs_gold_exact_by_split": {"train": 94.20, "validation": 94.12,
                                                   "test": 95.66},
                   "peer_accuracy_on_blind": "unmeasurable, there is no gold on blind",
                   "projection_by_c": proj},
               "rows_changed_vs_base": ch,
               "honest_summary": "Channel A is validated end to end and is worth about "
                                 f"{knn_gain:+.2f}. Channel B is a bracketed bet: +0.6 if the "
                                 "peer is as strong as this ensemble, ~-0.15 if it is much "
                                 "weaker. Central estimate 85.8-86.1. Cannot be verified: the "
                                 "Testing Phase closed 2026-08-03 12:00 UTC."},
              open(os.path.join(d, "meta.json"), "w"), indent=2)

    others = [p for root in (SUBMITTED, CANDIDATES) for p in sorted(root.iterdir()) if p.is_dir()]
    for prevdir in others:
        prev = prevdir.name
        q = os.path.join(prevdir, "prediction.zip")
        if prev == args.name or not os.path.exists(q):
            continue
        o = pd.read_csv(io.BytesIO(zipfile.ZipFile(q).read("prediction")))
        print(f"  vs {prev:28s} exact {100*(o.Prediction.values == pred).mean():5.1f}%  "
              f"QWK {fast_qwk(o.Prediction.values-1, pred-1)*100:.2f}")


if __name__ == "__main__":
    main()
