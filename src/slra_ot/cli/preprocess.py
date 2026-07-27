"""Turn BAREC sentences into the text variant the members expect, and cache it.

Writes data/processed/proc_{variant}_{split}.parquet with columns [ID, text, label19].
d3tok is slow, hence the cache.

    python -m slra_ot.cli.preprocess --variant d3tok
"""
import argparse
import time

import pandas as pd

from slra_ot.paths import PROCESSED, RAW_SPLITS, processed
from slra_ot.preprocessing import VARIANTS, transform_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    tfn = transform_fn(args.variant)

    for split in args.splits:
        df = pd.read_parquet(RAW_SPLITS[split])
        n = len(df)
        out_text, t0 = [], time.time()
        for i, s in enumerate(df["Sentence"].astype(str).tolist()):
            out_text.append(tfn(s))
            if args.variant == "d3tok" and (i + 1) % 5000 == 0:
                el = time.time() - t0
                print(f"  [{split}] {i+1}/{n}  {el:.0f}s  ({(i+1)/el:.0f} sent/s)", flush=True)
        out = pd.DataFrame({
            "ID": df["ID"].astype(str).values,
            "text": out_text,
            "label19": df["Readability_Level_19"].astype(int).values,
        })
        dst = processed(args.variant, split)
        out.to_parquet(dst)
        print(f"[{split}] {n} rows -> {dst}", flush=True)


if __name__ == "__main__":
    main()
