"""Pull the silver-labelled sentences out of BAREC-10M for pretraining.

BAREC-10M is public and CC-BY-SA-4.0, so it is allowed in the Open Track. It has ~552K
sentences with automatic 19-level labels, about 10x the gold training set, and it ships
pre-computed d3tok, which saves a ~2h CAMeL Tools run. Their d3tok format converts to the one
exactly on 70.5% of the overlapping sentences; the rest differ by analysis choice, not
formatting.

Important: BAREC-10M contains BAREC, so it contains dev and test. Pretraining on those,
even with silver labels, would leak into the sets used for early stopping and for
selection/calibration, which is the same mistake that sank sub03 arriving by another route.
So every sentence whose normalised text matches a dev or test sentence is dropped.

Blind sentences are not dropped. They have no gold label anywhere, nothing measured locally
uses them, and the corpus is public.

    python -m slra_ot.cli.build_silver
"""
import argparse
import json
import re
import unicodedata

import pandas as pd

from slra_ot.paths import BAREC10M_ANNOTATIONS, PROCESSED, RAW_SPLITS, silver

DIAC = re.compile(r"[ً-ْٰ]")     # harakat + dagger alef (keeps tatweel for now)
WS = re.compile(r"\s+")


def norm_key(s):
    return WS.sub(" ", unicodedata.normalize("NFKC", str(s))).strip()


def light_norm(s):
    return WS.sub(" ", str(s).replace("ـ", "")).strip()


def conv_d3tok(toks):
    """Their diacritised d3tok -> the split, undiacritised form the members were trained on."""
    s = " ".join(toks)
    s = s.replace("+ـ", "+ ").replace("ـ+", " +")   # clitic marker becomes a split token
    s = DIAC.sub("", s).replace("ـ", "")
    s = s.replace("ٱ", "ا")                          # alef wasla -> plain alef
    return WS.sub(" ", s).strip()


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    excl = set()
    for split in ["validation", "test"]:
        df = pd.read_parquet(RAW_SPLITS[split])
        excl |= {norm_key(s) for s in df.Sentence}
    print(f"excluding {len(excl)} dev+test sentences from the silver corpus", flush=True)

    files = [f for f in sorted(BAREC10M_ANNOTATIONS.rglob("*.json"))
             if "MACOSX" not in str(f)]
    print(f"{len(files)} annotation files", flush=True)

    rows, dropped, nolabel = [], 0, 0
    for n, f in enumerate(files):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        raws = d.get("raw_sents", [])
        rls = d.get("sents_RL", [])
        d3s = d.get("d3tok", [])
        for i, raw in enumerate(raws):
            if i >= len(rls) or i >= len(d3s):
                continue
            v = rls[i]
            try:
                lvl = int(v)
            except (TypeError, ValueError):
                nolabel += 1
                continue
            if not 1 <= lvl <= 19:
                nolabel += 1
                continue
            k = norm_key(raw)
            if k in excl:
                dropped += 1
                continue
            rows.append((f"s10m_{n}_{i}", light_norm(raw), conv_d3tok(d3s[i]), lvl))
        if (n + 1) % 4000 == 0:
            print(f"  {n+1}/{len(files)} files, {len(rows)} kept", flush=True)

    df = pd.DataFrame(rows, columns=["ID", "raw", "d3tok", "label19"])
    df = df[df.d3tok.str.len() > 0].drop_duplicates(subset="raw").reset_index(drop=True)
    print(f"\nkept {len(df)} | dropped {dropped} dev/test overlaps | {nolabel} unlabelled")
    print(df.label19.value_counts().sort_index().to_dict())

    for variant, col in [("raw", "raw"), ("d3tok", "d3tok")]:
        out = df[["ID", col, "label19"]].rename(columns={col: "text"})
        p = silver(variant)
        out.to_parquet(p, index=False)
        print(f"wrote {p}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
