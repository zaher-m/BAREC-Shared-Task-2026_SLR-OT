"""Fine-tune a decoder LLM as an ordinal regressor (LoRA + pooled head).

Same protocol as the _ad encoder members (train+dev for fitting, public test held out for
early stopping and cuts), so an LLM member needs no special handling in the pool.

    python -m slra_ot.cli.train_llm --model <hf-id> --tag allam7b_reg \
        --objective reg --variant raw --bs 8 --accum 4 --epochs 3
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from slra_ot.metrics import print_report, qwk
from slra_ot.modeling import OBJECTIVES, compute_loss, gpu_or_exit
from slra_ot.modeling_llm import LLMCollator, LLMRegressor, LLMSentDS, predict_scores
from slra_ot.paths import MEMBER_SCORES, MODELS, processed
from slra_ot.preprocessing import VARIANTS
from slra_ot.thresholds import QWKThresholdOptimizer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="raw", choices=VARIANTS)
    ap.add_argument("--objective", default="reg", choices=OBJECTIVES)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="trade compute for activation memory (unnecessary at these lengths)")
    ap.add_argument("--probe_steps", type=int, default=0,
                    help="if >0, run this many steps and report throughput, then exit")
    args = ap.parse_args()

    MEMBER_SCORES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = gpu_or_exit()

    tr = pd.read_parquet(processed(args.variant, "train"))
    dv = pd.read_parquet(processed(args.variant, "validation"))
    te = pd.read_parquet(processed(args.variant, "test"))
    tr = pd.concat([tr, dv], ignore_index=True)          # train+dev, test held out
    print(f"train={len(tr)}  holdout(test)={len(te)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"
    collate = LLMCollator(tok)
    tr_ds, te_ds = LLMSentDS(tr, tok, args.max_len), LLMSentDS(te, tok, args.max_len)
    tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=2,
                           pin_memory=True, collate_fn=collate)
    te_loader = DataLoader(te_ds, batch_size=args.eval_bs, num_workers=2, collate_fn=collate)

    model = LLMRegressor(args.model, args.objective, lora_r=args.lora_r,
                         grad_ckpt=args.grad_ckpt).to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params: {n_train/1e6:.1f}M", flush=True)

    head_ids = {id(p) for p in model.head.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.trainable_parameters() if id(p) not in head_ids],
         "lr": args.lr},
        {"params": list(model.head.parameters()), "lr": args.head_lr},
    ], weight_decay=0.01)
    steps = (len(tr_loader) // args.accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    if args.probe_steps:
        model.train()
        t0 = time.time()
        for i, batch in enumerate(tr_loader):
            if i >= args.probe_steps:
                break
            y = batch.pop("label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(args.objective, model(**batch), y)
            loss.backward()
            opt.step(); opt.zero_grad()
        el = time.time() - t0
        per_ep = el / args.probe_steps * len(tr_loader)
        print(f"[probe] {args.probe_steps} steps in {el:.1f}s -> "
              f"{args.probe_steps*args.bs/el:.1f} sent/s, ~{per_ep/60:.1f} min/epoch", flush=True)
        return

    best_qwk, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        opt.zero_grad()
        for step, batch in enumerate(tr_loader):
            y = batch.pop("label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(args.objective, model(**batch), y) / args.accum
            loss.backward()
            run += loss.item() * args.accum
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        s = predict_scores(model, te_loader, args.objective, device)
        q = qwk(te_ds.labels, np.clip(np.rint(s), 1, 19).astype(int))
        print(f"ep{ep} loss={run/len(tr_loader):.4f} holdout_QWK(naive)={q*100:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        if q > best_qwk:
            best_qwk, bad = q, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items() if "lora" in k or "head" in k}
        else:
            bad += 1
            if bad > args.patience:
                print("early stop", flush=True)
                break

    model.load_state_dict(best_state, strict=False)
    te_scores = predict_scores(model, te_loader, args.objective, device)
    opt_th = QWKThresholdOptimizer().fit(te_scores, te_ds.labels)
    print_report(f"{args.tag} HOLDOUT cal", te_ds.labels, opt_th.predict(te_scores))

    # same npz layout as the encoder members: dev slot mirrors the holdout, as for _ad
    np.savez(MEMBER_SCORES / f"scores_{args.tag}.npz",
             dev_ids=np.array(te_ds.ids), dev_scores=te_scores, dev_labels=np.array(te_ds.labels),
             test_ids=np.array(te_ds.ids), test_scores=te_scores,
             test_labels=np.array(te_ds.labels))
    ckpt = MODELS / args.tag
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt / "lora_head.pt")
    tok.save_pretrained(ckpt)
    meta = {"model": args.model, "variant": args.variant, "objective": args.objective,
            "max_len": args.max_len, "tag": args.tag, "kind": "llm_lora",
            "lora_r": args.lora_r,
            "dev_qwk_cal": qwk(te_ds.labels, opt_th.predict(te_scores)) * 100,
            "thresholds": opt_th.thresholds_.tolist()}
    for p in (MEMBER_SCORES / f"meta_{args.tag}.json", ckpt / "meta.json"):
        p.write_text(json.dumps(meta, indent=2))

    bl = pd.read_parquet(processed(args.variant, "blind"))
    bl_loader = DataLoader(LLMSentDS(bl, tok, args.max_len), batch_size=args.eval_bs,
                          num_workers=2, collate_fn=collate)
    np.save(MEMBER_SCORES / f"blind_scores_{args.tag}.npy",
            predict_scores(model, bl_loader, args.objective, device))
    print(f"saved scores_{args.tag}.npz, blind scores and checkpoint {ckpt}", flush=True)


if __name__ == "__main__":
    main()
