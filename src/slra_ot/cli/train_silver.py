"""Pretrain on the BAREC-10M silver labels, then fine-tune on gold.

By this point every post-processing idea had been measured and none of them worked
(document context, normalisation, shift alignment, global spread, per-sentence shrinkage,
external metadata, combiner variants). So the pool itself had to get better, and the one
resource left was BAREC-10M: ~439K sentences with automatic 19-level labels, eight times
the gold training set, public under CC-BY-SA-4.0.

Stage 1 trains on the silver labels. Stage 2 fine-tunes on gold. The head is kept between
stages: the target is the same 19 levels either way, so stage 2 starts from something that
already ranks readability instead of from a fresh head.

Stage 2 uses the clean protocol (train split, early stop on dev) so test stays usable for
selection and calibration, and build_silver already removed every dev/test sentence from
the silver corpus, so stage 1 cannot leak into them either.

    python -m slra_ot.cli.train_silver --tag arabertv2_emd_slv --objective emd --variant d3tok
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from slra_ot.metrics import print_report, qwk
from slra_ot.modeling import (OBJECTIVES, Encoder, PadCollator, SentDS, compute_loss,
                              gpu_or_exit, predict_scores)
from slra_ot.paths import MEMBER_SCORES, MODELS, processed, silver
from slra_ot.preprocessing import VARIANTS
from slra_ot.thresholds import QWKThresholdOptimizer


def loader(df, tok, max_len, bs, shuffle=False, workers=4):
    return DataLoader(SentDS(df, tok, max_len), batch_size=bs, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, collate_fn=PadCollator(tok))


def train_epochs(model, dl, objective, epochs, lr, device, warmup=0.06, tag="", eval_fn=None,
                 patience=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(warmup * total), total)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); run = 0.0
        for batch in dl:
            y = batch.pop("label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = compute_loss(objective, model(**batch), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += loss.item()
        msg = f"[{tag}] ep{ep} loss={run/len(dl):.4f} {time.time()-t0:.0f}s"
        if eval_fn is not None:
            q = eval_fn(model)
            msg += f" dev_QWK(naive)={q*100:.3f}"
            if q > best:
                best, bad = q, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        print(msg, flush=True)
        if patience is not None and eval_fn is not None and bad > patience:
            print("early stop", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aubmindlab/bert-base-arabertv2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="d3tok", choices=VARIANTS)
    ap.add_argument("--objective", default="emd", choices=OBJECTIVES)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--silver_bs", type=int, default=64)
    ap.add_argument("--silver_lr", type=float, default=3e-5)
    ap.add_argument("--silver_epochs", type=int, default=1)
    ap.add_argument("--silver_cap", type=int, default=0, help="subsample the silver corpus")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init_silver", action="store_true",
                    help="reuse cached silver-pretrained weights for this objective/variant "
                         "and skip stage 1 (~65 min saved)")
    args = ap.parse_args()

    MEMBER_SCORES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = gpu_or_exit()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Encoder(args.model, args.objective).to(device)

    # ---- stage 1: silver ----------------------------------------------------
    # Stage 1 takes ~65 min and is identical every time, so cache it. A member that only
    # changes gold hyper-parameters or the seed can reuse the weights and skip to stage 2.
    # The cache key has to include the model: bert-base and bert-large weights are not
    # interchangeable, and keying on objective/variant alone silently mixes them up.
    _mslug = args.model.replace("/", "-")
    cache = MODELS / f"_silverckpt_{_mslug}_{args.objective}_{args.variant}.pt"
    n_silver = -1   # -1 = reused cached stage-1 weights, corpus not re-read
    if args.init_silver and cache.exists():
        model.load_state_dict(torch.load(cache, map_location=device))
        print(f"[silver] loaded cached weights from {cache} (stage 1 skipped)", flush=True)
    else:
        sv = pd.read_parquet(silver(args.variant))
        if args.silver_cap:
            sv = sv.sample(n=min(args.silver_cap, len(sv)), random_state=args.seed)
        n_silver = len(sv)
        print(f"[silver] {len(sv)} sentences, {args.silver_epochs} epoch(s)", flush=True)
        train_epochs(model, loader(sv, tok, args.max_len, args.silver_bs, shuffle=True),
                     args.objective, args.silver_epochs, args.silver_lr, device, tag="silver")
        torch.save(model.state_dict(), cache)
        print(f"[silver] cached pretrained weights -> {cache}", flush=True)

    # ---- stage 2: gold, clean protocol --------------------------------------
    tr = pd.read_parquet(processed(args.variant, "train"))
    dv = pd.read_parquet(processed(args.variant, "validation"))
    te = pd.read_parquet(processed(args.variant, "test"))
    print(f"[gold] train={len(tr)} dev={len(dv)} (early stop) test={len(te)} (untouched)",
          flush=True)
    dv_dl = loader(dv, tok, args.max_len, 64)
    dv_labels = SentDS(dv, tok, args.max_len).labels

    def eval_dev(m):
        s = predict_scores(m, dv_dl, args.objective, device)
        return qwk(dv_labels, np.clip(np.rint(s), 1, 19).astype(int))

    train_epochs(model, loader(tr, tok, args.max_len, args.bs, shuffle=True),
                 args.objective, args.epochs, args.lr, device, tag="gold",
                 eval_fn=eval_dev, patience=args.patience)

    # ---- score + persist, same layout as every other member -----------------
    te_dl = loader(te, tok, args.max_len, 64)
    dv_ds, te_ds = SentDS(dv, tok, args.max_len), SentDS(te, tok, args.max_len)
    dv_s = predict_scores(model, dv_dl, args.objective, device)
    te_s = predict_scores(model, te_dl, args.objective, device)
    opt_th = QWKThresholdOptimizer().fit(dv_s, dv_ds.labels)
    print_report(f"{args.tag} DEV  cal", dv_ds.labels, opt_th.predict(dv_s))
    print_report(f"{args.tag} TEST cal", te_ds.labels, opt_th.predict(te_s))

    np.savez(MEMBER_SCORES / f"scores_{args.tag}.npz",
             dev_ids=np.array(dv_ds.ids), dev_scores=dv_s, dev_labels=np.array(dv_ds.labels),
             test_ids=np.array(te_ds.ids), test_scores=te_s, test_labels=np.array(te_ds.labels))
    ckpt = MODELS / args.tag
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt / "model.pt")
    tok.save_pretrained(ckpt)
    meta = {"model": args.model, "variant": args.variant, "objective": args.objective,
            "max_len": args.max_len, "tag": args.tag, "kind": "silver_pretrained",
            "silver_n": int(n_silver), "silver_epochs": args.silver_epochs,
            "dev_qwk_cal": qwk(dv_ds.labels, opt_th.predict(dv_s)) * 100,
            "thresholds": opt_th.thresholds_.tolist()}
    for p in (MEMBER_SCORES / f"meta_{args.tag}.json", ckpt / "meta.json"):
        p.write_text(json.dumps(meta, indent=2))

    bl = pd.read_parquet(processed(args.variant, "blind"))
    np.save(MEMBER_SCORES / f"blind_scores_{args.tag}.npy",
            predict_scores(model, loader(bl, tok, args.max_len, 64), args.objective, device))
    print(f"saved member {args.tag}", flush=True)


if __name__ == "__main__":
    main()
