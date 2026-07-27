"""Score a split with every trained member and cache the result.

Selecting an ensemble needs each member's score on the blind set. Re-running 50 checkpoints
every time would be pointless, so each member is scored once into
artifacts/member_scores/<split>_scores_<tag>.npy and everything after that is CPU work.

The preprocessed text is cached too, since d3tok has to run the disambiguator over every
sentence.

    python -m slra_ot.cli.score_split --split blind --all
    python -m slra_ot.cli.score_split --split blind --tags arabertv2_corn_ad marbert_soft_d3
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from slra_ot.modeling import Encoder, PadCollator, SentDS, gpu_or_exit, predict_scores
from slra_ot.paths import MEMBER_SCORES, MODELS, RAW_SPLITS, processed
from slra_ot.preprocessing import light_norm

ID_COL = {"blind": "Sentence ID"}     # every other split uses "ID"


def get_processed(split, variant):
    """Preprocessed text for a split, from the cache if it is there."""
    cache = processed(variant, split)
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_parquet(RAW_SPLITS[split])
    ids = df[ID_COL.get(split, "ID")].astype(str).tolist()
    texts = df["Sentence"].astype(str).tolist()
    if variant == "raw":
        proc = [light_norm(t) for t in texts]
    else:
        from slra_ot.preprocessing import build_d3tok
        tfn = build_d3tok()
        proc, n = [], len(texts)
        for i, t in enumerate(texts):
            proc.append(tfn(t))
            if (i + 1) % 2000 == 0:
                print(f"  [d3tok {split}] {i+1}/{n}", flush=True)
    label = (df["Readability_Level_19"].astype(int).values
             if "Readability_Level_19" in df else np.ones(len(df), int))
    out = pd.DataFrame({"ID": ids, "text": proc, "label19": label})
    out.to_parquet(cache)
    print(f"cached {cache}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="blind", choices=list(RAW_SPLITS))
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="every checkpoint under artifacts/models/")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.all:
        tags = sorted(p.parent.name for p in MODELS.glob("*/model.pt"))
    else:
        tags = args.tags
    if not tags:
        raise SystemExit("nothing to score: pass --tags or --all")

    todo = [t for t in tags
            if args.force or not (MEMBER_SCORES / f"{args.split}_scores_{t}.npy").exists()]
    print(f"{len(tags)} members, {len(todo)} to score on '{args.split}'", flush=True)
    if not todo:
        return

    metas = {t: json.loads((MODELS / t / "meta.json").read_text()) for t in todo}
    frames = {v: get_processed(args.split, v) for v in {m["variant"] for m in metas.values()}}

    ids_ref = None
    device = gpu_or_exit()
    for t in todo:
        meta = metas[t]
        df = frames[meta["variant"]]
        if ids_ref is None:
            ids_ref = df["ID"].astype(str).tolist()
            np.save(MEMBER_SCORES / f"{args.split}_ids.npy", np.array(ids_ref))
        tok = AutoTokenizer.from_pretrained(MODELS / t, trust_remote_code=True)
        model = Encoder(meta["model"], meta["objective"]).to(device)
        model.load_state_dict(torch.load(MODELS / t / "model.pt", map_location=device))
        loader = DataLoader(SentDS(df, tok, meta["max_len"]), batch_size=args.bs,
                            num_workers=2, collate_fn=PadCollator(tok))
        s = predict_scores(model, loader, meta["objective"], device)
        np.save(MEMBER_SCORES / f"{args.split}_scores_{t}.npy", s)
        del model
        torch.cuda.empty_cache()
        print(f"  scored {t:28s} mean={s.mean():.3f} sd={s.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
