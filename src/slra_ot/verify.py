"""Recompute every reported number from the committed artifacts and compare.

Run this first. It needs no GPU, no corpus and no trained weights, only what is committed
under artifacts/member_scores/, artifacts/logs/, configs/, experiments/ and submissions/.
If it prints anything other than "N/N checks passed", the docs are wrong, not the code.

    make verify              everything, ~3 min; the headline chain refits greedy selection
    make verify ARGS=--quick the same minus the slow refits, ~20 s

Determinism: nothing here resamples. Greedy selection, threshold fitting and the QWK
arithmetic are deterministic given the caches; floating-point order is fixed by the tag-sorted
member order in pool.txt files and meta weight dicts.
"""
import argparse
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd

from slra_ot import combiners as C
from slra_ot.members import aligned, blind_ids, blind_matrix, load_members, pinned_pool
from slra_ot.metrics import full_report
from slra_ot.paths import (CANDIDATES, EXPERIMENTS, LOGS, MEMBER_SCORES, SUBMITTED,
                           submission_dir)
from slra_ot.thresholds import (fit_thresholds, labels_from_thresholds,
                                prior_shrunk_thresholds, prior_thresholds)

N_MEMBERS = 76
DEGENERATE = "arabertv2_large_reg_ad"
TEST_ROWS, DEV_ROWS, BLIND_ROWS = 7286, 7310, 8077
TEST_LABEL_MEAN, DEV_LABEL_MEAN = 10.663739, 10.300137

# sub07: the headline configuration. CV estimate and blind score are context for the check
# names; the values verified here are the ones recomputable from the caches.
HEADLINE_INSAMPLE = 87.0178          # docs/data.md equivalence table, docs/reproduce.md
CONTAMINATION = (1.062, 18)          # docs/findings.md #2, analysis/contamination.py

UPLOADED = ["sub01_st_v5_transfer", "sub02_pool45_greedy", "sub03_bagged66",
            "sub04_cap30_prior50", "sub06_cap30_k035_silver", "sub07_clean_k035_silver",
            "sub08_clean48_bagged_k025", "sub09_clean47_k080"]
NEVER_UPLOADED = ["sub05_clean_prior65", "sub10_peerblend", "sub10_retrieval_only"]
# sub10_* are post-hoc blends over sub07, not weights-and-cuts systems
WEIGHTS_AND_CUTS = UPLOADED + ["sub05_clean_prior65"]


class Report:
    def __init__(self):
        self.n = 0
        self.failed = []

    def check(self, name, got, expected, tol=0.0, note=""):
        self.n += 1
        if isinstance(expected, float):
            ok = abs(got - expected) <= tol
            shown = f"{got:.4f} vs {expected:.4f}"
        else:
            ok = got == expected
            shown = f"{got} vs {expected}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<56s} {shown}  {note}")
        if not ok:
            self.failed.append((name, shown))


def read_spec(d):
    """Weights and cuts from a submission's meta.json / ensemble.json."""
    for name in ("meta.json", "ensemble.json"):
        p = d / name
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        w = m.get("weights") or m.get("members")
        if isinstance(w, dict):
            tags, weights = list(w), np.array(list(w.values()), float)
        elif isinstance(w, list) and "members" in m:
            tags, weights = list(m["members"]), np.array(m["weights"], float)
        else:
            return None
        return tags, weights / weights.sum(), np.array(m["thresholds"], float)
    return None


def uploaded_frame(d):
    """The recorded prediction: the bare CSV is the tracked record; the zip, when present
    locally, is the exact upload archive and is also validated."""
    csv = d / "prediction"
    if csv.exists():
        return pd.read_csv(csv)
    z = zipfile.ZipFile(d / "prediction.zip")
    return pd.read_csv(io.BytesIO(z.read("prediction")))


def check_member_records(r):
    ms = load_members(drop_degenerate=False)
    r.check("member score caches found", len(ms), N_MEMBERS)
    missing_meta = [m["tag"] for m in ms if not m["meta"]]
    missing_blind = [m["tag"] for m in ms
                     if not (MEMBER_SCORES / f"blind_scores_{m['tag']}.npy").exists()]
    r.check("every member has metadata", len(missing_meta), 0)
    r.check("every member has cached blind scores", len(missing_blind), 0)

    # stored calibrated dev QWK recomputes from the cached scores + the stored cuts
    worst, worst_tag = 0.0, ""
    for m in ms:
        if "dev_qwk_cal" not in m["meta"] or "thresholds" not in m["meta"]:
            continue
        th = np.array(m["meta"]["thresholds"], float)
        pred = labels_from_thresholds(m["npz"]["dev_scores"], th)
        got = full_report(m["npz"]["dev_labels"], pred)["QWK"]
        gap = abs(got - m["meta"]["dev_qwk_cal"])
        if gap > worst:
            worst, worst_tag = gap, m["tag"]
    r.check("stored per-member QWK recomputes (worst gap)", worst, 0.0, tol=1e-6,
            note=worst_tag)

    deg = next(m for m in ms if m["tag"] == DEGENERATE)
    r.check("the collapsed member really is constant (test sd)",
            float(deg["npz"]["test_scores"].std()), 0.0, tol=1e-12)

    clean = next(m for m in ms if m["tag"] == "arabertv2_emd_d3")["npz"]
    r.check("test split rows", len(clean["test_labels"]), TEST_ROWS)
    r.check("dev split rows", len(clean["dev_labels"]), DEV_ROWS)
    r.check("test label prior mean", float(np.mean(clean["test_labels"])),
            TEST_LABEL_MEAN, tol=1e-6)
    r.check("dev label prior mean", float(np.mean(clean["dev_labels"])),
            DEV_LABEL_MEAN, tol=1e-6)
    r.check("blind id vector rows", len(blind_ids()), BLIND_ROWS)


def check_pools(r):
    dirs = sorted(p for p in EXPERIMENTS.glob("exp0*") if (p / "pool.txt").exists())
    # exp001 pins its members inline (V5_MEMBERS) and exp014 is notes-only
    r.check("experiment directories with a pinned pool", len(dirs), 16)
    bad = []
    for d in dirs:
        tags = pinned_pool(d)
        if any(not (MEMBER_SCORES / f"scores_{t}.npz").exists() for t in tags):
            bad.append(d.name)
    r.check("every pinned tag has a score cache", len(bad), 0)
    # the three experiments that recorded their member list agree with the pin
    for exp, key, extra in [("exp002_combiner_bakeoff", "members", "clean subset"),
                            ("exp004_bagged_selection", "tags", "full pool"),
                            ("exp010_clean_pool", "clean_tags", "clean subset")]:
        rec = set(json.loads((EXPERIMENTS / exp / "results.json").read_text())[key])
        pin = set(pinned_pool(EXPERIMENTS / exp))
        ok = rec == pin or rec <= pin
        r.check(f"{exp} recorded member list inside its pin", ok, True, note=extra)


def check_submission_zips(r):
    bids = blind_ids()
    for name in UPLOADED + NEVER_UPLOADED:
        d = submission_dir(name)
        df = uploaded_frame(d)
        ok = (list(df.columns) == ["Sentence ID", "Prediction"]
              and list(df["Sentence ID"].astype(str)) == bids
              and bool(df.Prediction.between(1, 19).all())
              and not bool(df.Prediction.isna().any()))
        z = d / "prediction.zip"          # local-only; gitignored
        if z.exists():
            zf = zipfile.ZipFile(z)
            ok = ok and zf.namelist() == ["prediction"] and \
                zf.read("prediction") == (d / "prediction").read_bytes()
        r.check(f"{name}: record format, ids, range", ok, True)
    up = {p.name for p in SUBMITTED.iterdir() if p.is_dir()}
    cand = {p.name for p in CANDIDATES.iterdir() if p.is_dir()}
    r.check("submitted/ holds exactly the uploaded set", up, set(UPLOADED))
    r.check("candidates/ holds exactly the never-uploaded set", cand, set(NEVER_UPLOADED))


def check_submissions_rederive(r):
    """Every weights-and-cuts submission, re-derived from caches, row for row."""
    for name in WEIGHTS_AND_CUTS:
        d = submission_dir(name)
        tags, w, th = read_spec(d)
        X = blind_matrix(tags)
        pred = labels_from_thresholds(X @ w, th).astype(int)
        df = uploaded_frame(d)
        diff = int((df["Prediction"].values != pred).sum())
        r.check(f"{name}: rows differing from the zip", diff, 0,
                note=f"{len(pred)} rows, {len(tags)} members")


def check_cut_provenance(r):
    """The stored cuts are the named rule applied to the calibration scores."""
    RULES = {"sub02_pool45_greedy": ("free cuts, 0.02 grid",
                                     lambda s, y: fit_thresholds(s, y, rounds=8, grid=0.02)),
             "sub03_bagged66": ("free cuts, exact", fit_thresholds),
             "sub04_cap30_prior50": ("prior k=0.50", lambda s, y: prior_shrunk_thresholds(s, y, k=0.50)),
             "sub05_clean_prior65": ("prior k=0.65", lambda s, y: prior_shrunk_thresholds(s, y, k=0.65)),
             "sub06_cap30_k035_silver": ("prior k=0.35", lambda s, y: prior_shrunk_thresholds(s, y, k=0.35)),
             "sub07_clean_k035_silver": ("prior k=0.35", lambda s, y: prior_shrunk_thresholds(s, y, k=0.35)),
             "sub08_clean48_bagged_k025": ("prior k=0.25", lambda s, y: prior_shrunk_thresholds(s, y, k=0.25)),
             "sub09_clean47_k080": ("prior k=0.80", lambda s, y: prior_shrunk_thresholds(s, y, k=0.80))}
    for name, (label, rule) in RULES.items():
        d = submission_dir(name)
        tags, w, th = read_spec(d)
        ms = load_members(tags=tags, drop_degenerate=False)
        _, X, y = aligned(ms, "test")
        s = X @ w
        got = float(np.abs(np.sort(rule(s, y)) - np.sort(th)).max())
        # meta.json rounds weights to 5 decimals, which moves a cut in the 4th decimal
        r.check(f"{name}: cuts match rule '{label}' (max gap)", got, 0.0, tol=1e-3)
    # sub01's cuts were fitted in the Strict-Track project on dev; only its predictions,
    # not its cut rule, are verifiable here. prior_thresholds imported for the rule set
    # documented in evaluation.md.
    _ = prior_thresholds


def check_headline_forward(r):
    """sub07, forward from primary inputs: pool -> greedy -> cuts -> blind file.

    Nothing is taken from sub07's meta.json until the final comparison. The pool comes from
    exp017's pin, the weights from refitting greedy selection on the cached test scores, the
    cuts from the k=0.35 rule, the prediction from the cached blind scores.
    """
    tags = [t for t in pinned_pool(EXPERIMENTS / "exp017_clean_sweep")
            if not t.endswith(("_ad", "_ad2", "_ad3", "_ad4"))]
    r.check("headline pool: clean members in exp017's pin", len(tags), 47)
    ms = load_members(tags=tags, drop_degenerate=False)
    _, X, y = aligned(ms, "test")
    g = C.GreedyMultiset(rounds=30).fit(X, y)
    s = X @ g.weights_
    th = prior_shrunk_thresholds(s, y, k=0.35)
    insample = full_report(y, labels_from_thresholds(s, th))["QWK"]
    r.check("headline in-sample test QWK", insample, HEADLINE_INSAMPLE, tol=5e-4)

    Xb = blind_matrix(tags)
    pred = labels_from_thresholds(Xb @ g.weights_, th).astype(int)
    df = uploaded_frame(submission_dir("sub07_clean_k035_silver"))
    diff = int((df["Prediction"].values != pred).sum())
    r.check("headline blind file, rows differing from the uploaded zip", diff, 0,
            note=f"of {len(pred)}")

    picked = {tags[j]: round(float(w), 5) for j, w in enumerate(g.weights_) if w > 1e-9}
    meta = json.loads((submission_dir("sub07_clean_k035_silver") / "meta.json").read_text())
    r.check("headline refit picks the recorded 15 members", picked == meta["weights"], True)


def check_contamination(r):
    pat = re.compile(r"dev_QWK\(naive\)=([0-9.]+)")
    deltas = []
    for f in sorted(LOGS.glob("train_*ad*.log")):
        qs = [float(m) for m in pat.findall(f.read_text())]
        if len(qs) >= 2:
            deltas.append(max(qs) - float(np.mean(qs)))
    mean, n = CONTAMINATION
    r.check("contamination sample size (logs with epochs)", len(deltas), n)
    r.check("contamination: mean best-minus-mean-epoch QWK",
            float(np.mean(deltas)), mean, tol=5e-4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow greedy refit of the headline chain")
    args = ap.parse_args(argv)

    r = Report()
    print("member records")
    check_member_records(r)
    print("pinned pools")
    check_pools(r)
    print("submission files")
    check_submission_zips(r)
    print("submissions re-derived from caches")
    check_submissions_rederive(r)
    print("cut provenance")
    check_cut_provenance(r)
    print("contamination figure")
    check_contamination(r)
    if not args.quick:
        print("headline chain, forward from the pool")
        check_headline_forward(r)
    else:
        print("headline chain skipped (--quick)")

    print("=" * 78)
    if r.failed:
        print(f"{r.n - len(r.failed)}/{r.n} checks passed")
        for name, shown in r.failed:
            print(f"FAILED: {name} ({shown})")
        raise SystemExit(1)
    print(f"{r.n}/{r.n} checks passed")
    print("=" * 78)


if __name__ == "__main__":
    main()
