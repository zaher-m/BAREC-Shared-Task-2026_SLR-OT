"""Run a saved ensemble over new sentences and write the submission zip.

Reads the ensemble file written by select_ensemble plus each member's checkpoint, and
applies whichever preprocessing that member was trained on.

    python -m slra_ot.cli.infer --input blindtest.csv --ensemble artifacts/ensembles/v1.json \
        --out prediction
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from slra_ot.modeling import Encoder, PadCollator, SentDS, gpu_or_exit, predict_scores
from slra_ot.paths import ENSEMBLES, MODELS
from slra_ot.preprocessing import light_norm
from slra_ot.thresholds import labels_from_thresholds


def read_input(path, id_col, text_col):
    path = str(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith((".tsv", ".txt")):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    return df[id_col].astype(str).tolist(), df[text_col].astype(str).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--id-col", default="Sentence ID")
    ap.add_argument("--text-col", default="Sentence")
    ap.add_argument("--ensemble", default=str(ENSEMBLES / "ensemble.json"))
    ap.add_argument("--out", default="prediction")
    args = ap.parse_args()

    ens = json.loads(Path(args.ensemble).read_text())
    weights = np.asarray(ens["weights"], float); weights /= weights.sum()
    thresholds = np.asarray(ens["thresholds"], float)

    ids, texts = read_input(args.input, args.id_col, args.text_col)

    metas = {tag: json.loads((MODELS / tag / "meta.json").read_text()) for tag in ens["members"]}
    variants = {m["variant"] for m in metas.values()}
    proc = {}
    if "raw" in variants:
        proc["raw"] = [light_norm(t) for t in texts]
    if "d3tok" in variants:
        from slra_ot.preprocessing import build_d3tok
        tfn = build_d3tok()
        proc["d3tok"] = [tfn(t) for t in texts]

    device = gpu_or_exit()
    member_scores = []
    for tag in ens["members"]:
        meta = metas[tag]
        tok = AutoTokenizer.from_pretrained(MODELS / tag, trust_remote_code=True)
        model = Encoder(meta["model"], meta["objective"]).to(device)
        model.load_state_dict(torch.load(MODELS / tag / "model.pt", map_location=device))
        base = pd.DataFrame({"ID": ids, "text": proc[meta["variant"]], "label19": [1] * len(ids)})
        loader = DataLoader(SentDS(base, tok, meta["max_len"]), batch_size=64, num_workers=0,
                            collate_fn=PadCollator(tok))
        member_scores.append(predict_scores(model, loader, meta["objective"], device))
        del model
        torch.cuda.empty_cache()
        print(f"scored {tag}")

    ens_score = np.vstack(member_scores).T @ weights
    pred = labels_from_thresholds(ens_score, thresholds).astype(int)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Sentence ID": ids, "Prediction": pred}).to_csv(out, index=False)
    with zipfile.ZipFile(str(out) + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out, arcname="prediction")
    print(f"wrote {out} and {out}.zip  ({len(ids)} rows)")


if __name__ == "__main__":
    main()
