"""Encoder member: backbone, mean pooling, one head, six objectives.

  reg  : plain regression, MSE on the 1..19 scale (best single baseline in the BAREC paper)
  corn : CORN ordinal regression, 18 rank-consistent logits
  soft : KL to a soft target peaked at the gold level
  wkl  : cross-entropy plus a soft-QWK penalty
  emd  : squared earth-mover distance between predicted and one-hot CDFs
  oll  : ordinal log-loss, weighted by distance

All six return a continuous level in 1..19, so any member can go straight into the same
combiner and the same thresholding.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

from slra_ot.metrics import N_CLASSES

OBJECTIVES = ("reg", "corn", "soft", "wkl", "emd", "oll")


def gpu_or_exit():
    """The CUDA device, or an explanation of what can still be run without one."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device found. Training a member and scoring a split both need a GPU.\n"
            "Everything downstream of the cached member scores runs on CPU: the experiments\n"
            "under experiments/, the diagnostics under analysis/, slra_ot.cli.select_ensemble,\n"
            "slra_ot.cli.build_submission and slra_ot.cli.verify."
        )
    return "cuda"


class SentDS(Dataset):
    def __init__(self, df, tok, max_len):
        self.texts = df["text"].astype(str).tolist()
        self.ids = df["ID"].astype(str).tolist()
        self.labels = df["label19"].astype(int).tolist()
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        # No padding here, the collator pads each batch to its own longest sequence.
        # Sentences are short (p95 = 40 tokens, max 117) against max_len 160, so fixed
        # padding was wasting ~90% of the compute. Masked attention and mask-weighted mean
        # pooling mean the logits come out the same either way. About 3x faster.
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len)
        item = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
        item["label"] = self.labels[i] - 1  # 0..18
        return item


class PadCollator:
    """Pad each batch to its own longest sequence.

    Has to be a class, not a closure, or the DataLoader workers cannot pickle it.
    """

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, features):
        labels = torch.tensor([f.pop("label") for f in features], dtype=torch.long)
        batch = self.tok.pad(features, padding=True, return_tensors="pt")
        batch["label"] = labels
        return batch


class Encoder(nn.Module):
    def __init__(self, name, objective, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name, trust_remote_code=True)
        h = self.backbone.config.hidden_size
        self.objective = objective
        self.drop = nn.Dropout(dropout)
        out = 1 if objective == "reg" else (N_CLASSES - 1 if objective == "corn" else N_CLASSES)
        self.head = nn.Linear(h, out)

    def forward(self, **batch):
        model_in = {k: v for k, v in batch.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = self.backbone(**model_in)
        last = out.last_hidden_state
        mask = model_in["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # mean pooling
        return self.head(self.drop(pooled))


# ---- ordinal-regression objective (18 logits) -------------------------------

def corn_loss(logits, y, num_classes):
    """CORN loss (Shi et al.), written out here to avoid pulling in coral_pytorch."""
    losses, n_terms = 0.0, 0
    for k in range(num_classes - 1):
        mask = y > (k - 1) if k > 0 else torch.ones_like(y, dtype=torch.bool)  # subset y>=k
        if mask.sum() == 0:
            continue
        lg = logits[mask, k]
        target = (y[mask] > k).float()
        losses = losses + nn.functional.binary_cross_entropy_with_logits(
            lg, target, reduction="sum")
        n_terms += mask.sum().item()
    return losses / max(n_terms, 1)


def corn_scores(logits):
    """Expected level from CORN logits, via the cumulative sigmoids."""
    probs = torch.sigmoid(logits)
    cum = torch.cumprod(probs, dim=1)  # P(y>k)
    return 1.0 + cum.sum(dim=1)


# ---- classification-head objectives (19 logits) ----------------------------

_LEVELS = torch.arange(N_CLASSES).float()          # 0..18
_WMAT = (torch.arange(N_CLASSES).float()[:, None]
         - torch.arange(N_CLASSES).float()[None, :]) ** 2 / (N_CLASSES - 1) ** 2


def soft_targets(y, T=2.0):
    lv = _LEVELS.to(y.device)
    d2 = (lv[None, :] - y[:, None].float()) ** 2
    return torch.softmax(-d2 / T, dim=1)           # (B,19) unimodal peaked at y


def soft_ce_loss(logits, y, T=2.0):
    logp = torch.log_softmax(logits, dim=1)
    return nn.functional.kl_div(logp, soft_targets(y, T), reduction="batchmean")


def wkl_loss(logits, y, lam=0.5):
    """CE plus a soft QWK term (de la Torre). Minimising num/den is roughly 1 - kappa."""
    ce = nn.functional.cross_entropy(logits, y)
    p = torch.softmax(logits, dim=1)
    W = _WMAT.to(logits.device)
    num = (W[y] * p).sum()                          # soft observed disagreement
    hist = torch.bincount(y, minlength=N_CLASSES).float()
    mean_p = p.sum(0)
    den = (W * torch.outer(hist, mean_p)).sum() / len(y) + 1e-6
    return ce + lam * (num / den)


def emd_loss(logits, y):
    p = torch.softmax(logits, dim=1)
    oh = nn.functional.one_hot(y, N_CLASSES).float()
    return ((p.cumsum(1) - oh.cumsum(1)) ** 2).sum(1).mean()


def oll_loss(logits, y, alpha=1.5):
    p = torch.softmax(logits, dim=1).clamp(1e-6, 1 - 1e-6)
    lv = _LEVELS.to(y.device)
    d = (lv[None, :] - y[:, None].float()).abs() ** alpha
    return (-torch.log(1 - p) * d).sum(1).mean()


def expected_scores(logits):
    """Expected level from a 19-way head."""
    p = torch.softmax(logits, dim=1)
    lv = (torch.arange(N_CLASSES).float() + 1.0).to(logits.device)
    return (p * lv[None, :]).sum(1)


def compute_loss(objective, logits, y):
    if objective == "reg":  return nn.functional.mse_loss(logits.squeeze(-1), y.float() + 1.0)
    if objective == "corn": return corn_loss(logits, y, N_CLASSES)
    if objective == "soft": return soft_ce_loss(logits, y)
    if objective == "wkl":  return wkl_loss(logits, y)
    if objective == "emd":  return emd_loss(logits, y)
    if objective == "oll":  return oll_loss(logits, y)
    raise ValueError(objective)


def predict_scores(model, loader, objective, device):
    """One continuous score per row, in loader order."""
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "label"}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            if objective == "reg":
                s = logits.squeeze(-1).float()
            elif objective == "corn":
                s = corn_scores(logits.float())
            else:
                s = expected_scores(logits.float())
            scores.append(s.cpu().numpy())
    return np.concatenate(scores)
