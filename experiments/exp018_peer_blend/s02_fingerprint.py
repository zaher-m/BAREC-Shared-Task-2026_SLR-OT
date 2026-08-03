"""exp018 step 2: are the BAREC-10M labels on blind sentences gold, or model output?

This decides whether the blend is worth doing at all, and at what weight.

sents_RL agrees with gold at QWK 97.96 / 95% exact on dev+test. Two explanations:
  (a) it is the human BAREC label copied through their pipeline, losing ~5% to sentence
      splitting and de-duplication mismatches;
  (b) it is a model prediction, and the model was trained on dev+test so it memorised them.

Under (a) the covered blind sentences carry gold labels and the blend is nearly a
replacement. Under (b) they are a peer model's out-of-sample predictions and the blend is
variance reduction at a modest weight.

The corpus is one JSON per source document, so the document of each sentence is known, which
gives three groups inside the same corpus sharing the same label field:

  G  documents whose sentences match BAREC train/dev/test  -> gold-derived
  B  documents whose sentences match the blind set         -> the question
  U  documents matching no BAREC split                     -> model output

If B looks like G the labels are gold; if it looks like U they are predictions. What separates
a human label set from model predictions here:
  - how much mass sits on levels 1 and 19, since models under-predict the extremes
  - label entropy and how many levels get used at all
  - whether G and B documents ever appear in the same file

Both are integers, so the distribution shape is all there is to go on.
"""
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from slra_ot.paths import REGENERATED  # noqa: E402

# re-runs write here, never over the committed record
OUT_DIR = os.path.join(str(REGENERATED), os.path.basename(HERE))
os.makedirs(OUT_DIR, exist_ok=True)

from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.paths import BAREC10M_ANNOTATIONS, RAW_SPLITS  # noqa: E402

SRC = BAREC10M_ANNOTATIONS
WS = re.compile(r"\s+")


def key(s):
    return WS.sub(" ", unicodedata.normalize("NFKC", str(s)).replace("ـ", "")).strip()


def hist_stats(labels, name):
    c = Counter(labels)
    n = len(labels)
    p = np.array([c.get(k, 0) / n for k in range(1, 20)])
    ent = -np.sum([x * np.log(x) for x in p if x > 0])
    return {"name": name, "n": n, "mean": float(np.mean(labels)), "sd": float(np.std(labels)),
            "p_lvl1": float(p[0]), "p_lvl19": float(p[18]),
            "p_extreme": float(p[0] + p[18]), "levels_used": int((p > 0).sum()),
            "entropy": float(ent), "p": p}


def main():
    # --- reference label sets from the gold splits ---------------------------
    gold = {}
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(RAW_SPLITS[split])
        col = "Readability_Level_19" if "Readability_Level_19" in df.columns else df.columns[-1]
        for s, l in zip(df.Sentence, df[col]):
            gold[key(s)] = int(l)
    blind = {key(s) for s in pd.read_parquet(RAW_SPLITS["blind"]).Sentence}
    print(f"gold sentences {len(gold)} | blind sentences {len(blind)}")

    files = [f for f in glob.glob(os.path.join(SRC, "**", "*.json"), recursive=True)
             if "MACOSX" not in f]
    print(f"{len(files)} annotation files\n")

    # --- classify every file by which populations its sentences hit ----------
    G, B, U = [], [], []
    pairs_gold = []                     # (gold label, sents_RL) on gold-matched sentences
    file_rows = []
    mixed = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        raws, rls = d.get("raw_sents", []), d.get("sents_RL", [])
        ng = nb = nu = 0
        for i, raw in enumerate(raws):
            if i >= len(rls):
                break
            lvl = rls[i]
            if not isinstance(lvl, int) or not (1 <= lvl <= 19):
                continue
            k = key(raw)
            if k in gold:
                ng += 1; G.append(lvl); pairs_gold.append((gold[k], lvl))
            elif k in blind:
                nb += 1; B.append(lvl)
            else:
                nu += 1; U.append(lvl)
        if ng and nb:
            mixed += 1
        if ng or nb or nu:
            file_rows.append({"file": os.path.basename(f), "gold": ng, "blind": nb, "unk": nu})

    print(f"sentences by population:  G(gold-matched) {len(G)}  "
          f"B(blind-matched) {len(B)}  U(unmatched) {len(U)}")
    print(f"files containing BOTH gold and blind sentences: {mixed}")

    fr = pd.DataFrame(file_rows)
    pure_g = ((fr.gold > 0) & (fr.blind == 0) & (fr.unk == 0)).sum()
    pure_b = ((fr.blind > 0) & (fr.gold == 0) & (fr.unk == 0)).sum()
    pure_u = ((fr.unk > 0) & (fr.gold == 0) & (fr.blind == 0)).sum()
    print(f"pure files:  gold-only {pure_g}  blind-only {pure_b}  unmatched-only {pure_u}")
    print("  a blind-only file means the whole source document is held out, consistent "
          "with BAREC's document-level split")

    # --- the decisive comparison -------------------------------------------
    sg, sb, su = hist_stats(G, "G gold-matched"), hist_stats(B, "B blind"), hist_stats(U, "U unmatched")
    print(f"\n{'population':>16s} {'n':>7s} {'mean':>6s} {'sd':>6s} {'p(lvl1)':>8s} "
          f"{'p(lvl19)':>9s} {'p(extreme)':>11s} {'levels':>7s} {'entropy':>8s}")
    for s in (sg, sb, su):
        print(f"{s['name']:>16s} {s['n']:7d} {s['mean']:6.2f} {s['sd']:6.2f} "
              f"{s['p_lvl1']:8.4f} {s['p_lvl19']:9.4f} {s['p_extreme']:11.4f} "
              f"{s['levels_used']:7d} {s['entropy']:8.4f}")

    # gold reference straight from the gold parquets, for scale
    gl = list(gold.values())
    s0 = hist_stats(gl, "BAREC gold")
    print(f"{s0['name']:>16s} {s0['n']:7d} {s0['mean']:6.2f} {s0['sd']:6.2f} "
          f"{s0['p_lvl1']:8.4f} {s0['p_lvl19']:9.4f} {s0['p_extreme']:11.4f} "
          f"{s0['levels_used']:7d} {s0['entropy']:8.4f}")

    # distances between histograms: does B sit with G or with U?
    def tv(a, b):
        return 0.5 * float(np.abs(a - b).sum())
    print(f"\ntotal-variation distance between label histograms:")
    print(f"  B vs G  {tv(sb['p'], sg['p']):.4f}")
    print(f"  B vs U  {tv(sb['p'], su['p']):.4f}")
    print(f"  G vs U  {tv(sg['p'], su['p']):.4f}")
    print(f"  G vs BAREC gold  {tv(sg['p'], s0['p']):.4f}   (should be ~0 if G is gold)")

    # --- how faithful is sents_RL to gold where both are visible? -----------
    a = np.array([x for x, _ in pairs_gold]); b = np.array([y for _, y in pairs_gold])
    print(f"\non the {len(a)} gold-matched sentences: sents_RL vs gold "
          f"QWK {fast_qwk(a-1, b-1)*100:.3f}  exact {100*(a==b).mean():.2f}%  "
          f"MAE {np.abs(a-b).mean():.4f}")
    off = Counter((b - a).tolist())
    print("  signed error histogram (sents_RL - gold), most common:",
          sorted(off.items(), key=lambda kv: -kv[1])[:7])

    json.dump({"n_G": len(G), "n_B": len(B), "n_U": len(U), "mixed_files": int(mixed),
               "pure_gold_files": int(pure_g), "pure_blind_files": int(pure_b),
               "pure_unmatched_files": int(pure_u),
               "stats": {s["name"]: {k: v for k, v in s.items() if k != "p"}
                         for s in (sg, sb, su, s0)},
               "tv_B_G": tv(sb["p"], sg["p"]), "tv_B_U": tv(sb["p"], su["p"]),
               "tv_G_U": tv(sg["p"], su["p"]),
               "sentsRL_vs_gold_qwk": float(fast_qwk(a - 1, b - 1) * 100),
               "sentsRL_vs_gold_exact": float((a == b).mean())},
              open(os.path.join(OUT_DIR, "s02_fingerprint.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
