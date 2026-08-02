"""Fine-tune one encoder with one objective on BAREC sentences.

Three protocols. Which one a member used decides what it can be used for later:

  default    train on the BAREC train split, early-stop on dev. Test is never touched, not
             by training and not by model selection, so fitting the ensemble on test is
             fine. These are the clean members.
  --alldata  train on train+dev (+13% data), early-stop on test. Better models, but test is
             then contaminated for selection and calibration, worth 1.06 QWK of inflation
             (exp010). These are the _ad members.
  --fold K   one document-grouped fold over all 69,441 labelled sentences. Open Track only,
             since it needs the public test labels.

Saves the dev/test scores for ensembling, a checkpoint and a small metadata file.

    python -m slra_ot.cli.train_encoder --model <hf-id> --tag arabertv2_emd_c2 \
        --variant d3tok --objective emd --seed 777
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from slra_ot.metrics import N_CLASSES, print_report, qwk
from slra_ot.modeling import (OBJECTIVES, Encoder, PadCollator, SentDS, compute_loss,
                              gpu_or_exit, predict_scores)
from slra_ot.paths import MEMBER_SCORES, MODELS, processed
from slra_ot.preprocessing import VARIANTS
from slra_ot.thresholds import QWKThresholdOptimizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="short id used in output filenames")
    ap.add_argument("--variant", default="d3tok", choices=VARIANTS)
    ap.add_argument("--objective", default="reg", choices=OBJECTIVES)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--upsample", action="store_true", help="inverse-freq (sqrt) weighted sampling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alldata", action="store_true",
                    help="train on train+dev (+13%% data); hold out TEST for early stopping "
                         "and calibration")
    ap.add_argument("--fold", type=int, default=None,
                    help="document-grouped fold index over ALL 69,441 labelled sentences")
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    MEMBER_SCORES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = gpu_or_exit()

    tr = pd.read_parquet(processed(args.variant, "train"))
    dv = pd.read_parquet(processed(args.variant, "validation"))
    te = pd.read_parquet(processed(args.variant, "test"))

    if args.fold is not None:
        from slra_ot.folds import fold_frames
        tr, dv = fold_frames(args.variant, args.fold, args.n_folds)
        te = dv
        print(f"[fold {args.fold}/{args.n_folds}] train={len(tr)} holdout={len(dv)} "
              f"(document-grouped over all 69,441)", flush=True)
    elif args.alldata:
        tr = pd.concat([tr, dv], ignore_index=True)
        dv = te
        print(f"[alldata] train={len(tr)} (train+dev), holdout=test={len(te)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tr_ds = SentDS(tr, tok, args.max_len)
    dv_ds = SentDS(dv, tok, args.max_len)
    te_ds = SentDS(te, tok, args.max_len)
    collate = PadCollator(tok)

    if args.upsample:
        labels = np.array(tr_ds.labels)
        freq = np.bincount(labels, minlength=N_CLASSES).astype(float)
        w_per_class = 1.0 / np.sqrt(np.clip(freq, 1, None))  # inverse freq, damped by sqrt
        sample_w = w_per_class[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                        len(labels), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, num_workers=4,
                               pin_memory=True, collate_fn=collate)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=4,
                               pin_memory=True, collate_fn=collate)
    dv_loader = DataLoader(dv_ds, batch_size=64, num_workers=4, pin_memory=True, collate_fn=collate)
    te_loader = DataLoader(te_ds, batch_size=64, num_workers=4, pin_memory=True, collate_fn=collate)

    model = Encoder(args.model, args.objective).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    best_qwk, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for batch in tr_loader:
            y = batch.pop("label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch)
                loss = compute_loss(args.objective, logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += loss.item()
        dv_scores = predict_scores(model, dv_loader, args.objective, device)
        dv_pred = np.clip(np.rint(dv_scores), 1, 19).astype(int)
        q = qwk(dv_ds.labels, dv_pred)
        print(f"ep{ep} loss={run/len(tr_loader):.4f} dev_QWK(naive)={q*100:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        if q > best_qwk:
            best_qwk, bad = q, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > args.patience:
                print("early stop", flush=True); break

    model.load_state_dict(best_state)
    dv_scores = predict_scores(model, dv_loader, args.objective, device)
    te_scores = predict_scores(model, te_loader, args.objective, device)

    # fit cuts on the holdout and report both splits. The ensemble refits its own cuts.
    opt_th = QWKThresholdOptimizer().fit(dv_scores, dv_ds.labels)
    dv_pred = opt_th.predict(dv_scores); te_pred = opt_th.predict(te_scores)
    print_report(f"{args.tag} DEV  cal", dv_ds.labels, dv_pred)
    print_report(f"{args.tag} TEST cal", te_ds.labels, te_pred)

    np.savez(MEMBER_SCORES / f"scores_{args.tag}.npz",
             dev_ids=np.array(dv_ds.ids), dev_scores=dv_scores, dev_labels=np.array(dv_ds.labels),
             test_ids=np.array(te_ds.ids), test_scores=te_scores,
             test_labels=np.array(te_ds.labels))
    ckpt_dir = MODELS / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / "model.pt")
    tok.save_pretrained(ckpt_dir)
    meta = {"model": args.model, "variant": args.variant, "objective": args.objective,
            "max_len": args.max_len, "tag": args.tag,
            "dev_qwk_cal": qwk(dv_ds.labels, dv_pred) * 100,
            "thresholds": opt_th.thresholds_.tolist()}
    for p in (MEMBER_SCORES / f"meta_{args.tag}.json", ckpt_dir / "meta.json"):
        p.write_text(json.dumps(meta, indent=2))
    print(f"saved scores_{args.tag}.npz and checkpoint {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
