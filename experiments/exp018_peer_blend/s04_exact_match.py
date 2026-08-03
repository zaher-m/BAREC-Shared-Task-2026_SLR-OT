"""exp018 step 4: do any blind sentences appear word-for-word in the public gold splits?

This falls out of step 3. sents_RL copies the human label wherever BAREC has one, so the
corpus contains text/label pairs. But BAREC-10M is not needed for that: BAREC train,
validation and public test are public and carry gold labels themselves. If a blind sentence
appears verbatim in one of them, its label is public information. Nothing held out is
involved, it is just nearest-neighbour lookup at distance zero.

Short Arabic sentences repeat across documents (chapter headings, Qur'anic verses, proverbs),
so duplicates across splits are expected. That duplication is in the corpus, not something
introduced.

Precision matters more than recall, because a wrong override costs more than a missed one:
the ensemble is already exactly right ~40% of the time and within +-1 ~71% of the time. So
the lookup is gated:
  - normalised exact text match only (NFKC, tatweel stripped, whitespace collapsed)
  - the matched text must have a consistent label across all its public occurrences
    (spread <= 1 level), otherwise it is refused
  - very short strings are refused, since those repeat for reasons unrelated to readability
"""
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from slra_ot.paths import REGENERATED  # noqa: E402

# re-runs write here, never over the committed record
OUT_DIR = os.path.join(str(REGENERATED), os.path.basename(HERE))
os.makedirs(OUT_DIR, exist_ok=True)

from slra_ot import combiners as C  # noqa: E402
from slra_ot.members import aligned, blind_matrix, pinned_members  # noqa: E402
from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.paths import RAW_SPLITS  # noqa: E402
from slra_ot.thresholds import labels_from_thresholds  # noqa: E402

WS = re.compile(r"\s+")
AD = ("_ad", "_ad2", "_ad3", "_ad4")
MIN_WORDS = 3


def key(s):
    return WS.sub(" ", unicodedata.normalize("NFKC", str(s)).replace("ـ", "")).strip()


def public_lut(splits):
    """normalised text -> (mean gold label, spread, count) over the given splits."""
    acc = defaultdict(list)
    for split in splits:
        df = pd.read_parquet(RAW_SPLITS[split])
        col = [c for c in df.columns if "19" in c] or [df.columns[-1]]
        for s, l in zip(df.Sentence, df[col[0]]):
            acc[key(s)].append(int(l))
    return {k: (float(np.mean(v)), int(max(v) - min(v)), len(v)) for k, v in acc.items()}


def main():
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    ids, Xt, yt = aligned(members, "test")
    Xb = blind_matrix(tags)
    W = C.GreedyMultiset(rounds=30).fit(Xt, yt).weights_
    s_test, s_blind = Xt @ W, Xb @ W
    th = C.prior_shrunk_thresholds(s_test, yt, k=0.35)
    p_blind = labels_from_thresholds(s_blind, th)

    bl = pd.read_parquet(RAW_SPLITS["blind"])
    print(f"blind set: {len(bl)} sentences")

    # ---- how much of the blind set matches public gold word for word? --------
    for splits in (["train"], ["train", "validation"], ["train", "validation", "test"]):
        lut = public_lut(splits)
        hit = np.zeros(len(bl), bool)
        lab = np.zeros(len(bl))
        amb = short = 0
        for i, s in enumerate(bl.Sentence.values):
            k = key(s)
            if k not in lut:
                continue
            m, spread, n = lut[k]
            if len(k.split()) < MIN_WORDS:
                short += 1
                continue
            if spread > 1:
                amb += 1
                continue
            hit[i], lab[i] = True, m
        name = "+".join(splits)
        print(f"\npublic splits {name:28s} -> {hit.sum():5d} verbatim blind matches "
              f"({100*hit.mean():5.2f}%)   refused: {amb} ambiguous, {short} too short")
        if hit.sum():
            base = p_blind[hit]
            print(f"   the current prediction vs the public gold label on those rows:")
            print(f"     exact  {100*(base == np.rint(lab[hit])).mean():5.1f}%   "
                  f"within+-1 {100*(np.abs(base - lab[hit]) <= 1).mean():5.1f}%   "
                  f"MAE {np.abs(base - lab[hit]).mean():.3f}   "
                  f"QWK {fast_qwk(np.rint(lab[hit]).astype(int)-1, base-1)*100:.2f}")
            print(f"     mean gold {lab[hit].mean():.2f}  mean pred {base.mean():.2f}")
            sse_now = float(np.sum((base - lab[hit]) ** 2))
            print(f"     squared error on these rows now {sse_now:.0f}; "
                  f"an exact override would make it ~0")
            # rough: what share of the total blind squared error is this? Estimate the
            # total from the per-row MSE on test, with a bit added since blind is harder.
            mse_test = float(np.mean((labels_from_thresholds(s_test, th) - yt) ** 2))
            tot = mse_test * len(bl) * 1.08
            print(f"     that is ~{100*sse_now/tot:.2f}% of estimated total blind squared "
                  f"error -> roughly {100*(1-0.855)*(sse_now/tot):.2f} QWK points recoverable")
            best = (name, hit.copy(), lab.copy())

    # ---- dry run on the public test split ------------------------------------
    # Build the lookup from train+dev only, apply it to public test, measure the real gain.
    # Same mechanism as on blind, except here the gold labels exist.
    print("\n" + "=" * 72)
    print("DRY RUN on the public test split, lookup built from train+validation only")
    lut = public_lut(["train", "validation"])
    idcol = None
    te_raw = pd.read_parquet(RAW_SPLITS["test"])
    idcol = [c for c in te_raw.columns if "id" in c.lower()][0]
    tmap = dict(zip(te_raw[idcol].astype(str), te_raw.Sentence))
    te_texts = [tmap[str(i)] for i in ids]

    p_test = labels_from_thresholds(s_test, th)
    hit = np.zeros(len(ids), bool); lab = np.zeros(len(ids))
    for i, s in enumerate(te_texts):
        k = key(s)
        if k not in lut or len(k.split()) < MIN_WORDS:
            continue
        m, spread, n = lut[k]
        if spread > 1:
            continue
        hit[i], lab[i] = True, m
    print(f"public-test rows with a verbatim train/dev match: {hit.sum()} "
          f"({100*hit.mean():.2f}%)")
    if hit.sum():
        print(f"  lookup label vs gold on those rows: exact "
              f"{100*(np.rint(lab[hit]) == yt[hit]).mean():.1f}%  "
              f"MAE {np.abs(lab[hit] - yt[hit]).mean():.3f}")
        print(f"  prediction vs gold on those rows: exact "
              f"{100*(p_test[hit] == yt[hit]).mean():.1f}%  "
              f"MAE {np.abs(p_test[hit] - yt[hit]).mean():.3f}")
        q0 = fast_qwk(yt - 1, p_test - 1) * 100
        p2 = p_test.copy(); p2[hit] = np.clip(np.rint(lab[hit]), 1, 19).astype(int)
        q1 = fast_qwk(yt - 1, p2 - 1) * 100
        print(f"\n  TEST QWK without override {q0:.3f}")
        print(f"  TEST QWK with    override {q1:.3f}   ({q1-q0:+.3f})")
        # and a soft version, in case hard override is too aggressive. The retrieved label
        # is on the level scale, so map it back through the linear fit before blending.
        c0, c1 = np.polyfit(s_test, yt, 1)
        for a in (0.5, 0.75, 1.0):
            tgt = (lab[hit] - c1) / c0
            sm = s_test.copy(); sm[hit] = (1 - a) * s_test[hit] + a * tgt
            q = fast_qwk(yt - 1, labels_from_thresholds(sm, th) - 1) * 100
            print(f"  soft override alpha={a:.2f} -> TEST QWK {q:.3f} ({q-q0:+.3f})")

    json.dump({"min_words": MIN_WORDS,
               "blind_verbatim_matches": int(best[1].sum()),
               "dry_run_test_hits": int(hit.sum())},
              open(os.path.join(OUT_DIR, "s04_exact_match.json"), "w"), indent=2)
    np.save(os.path.join(OUT_DIR, "blind_lookup_hit.npy"), best[1])
    np.save(os.path.join(OUT_DIR, "blind_lookup_label.npy"), best[2])


if __name__ == "__main__":
    main()
