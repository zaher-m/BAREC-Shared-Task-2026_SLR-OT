"""Decoder-LLM member: causal trunk with LoRA and a pooled ordinal head.

The encoder pool is all BERT-scale models (110M-370M) trained on much the same Arabic text,
so they agree with each other and the ensemble stops improving. The idea here was that a 7B
Arabic LLM has a different enough prior to be worth more than another correlated BERT.

Trained but never finished in time to make the pool.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

from slra_ot.metrics import N_CLASSES
from slra_ot.modeling import corn_scores, expected_scores


class LLMSentDS(Dataset):
    def __init__(self, df, tok, max_len):
        self.texts = df["text"].astype(str).tolist()
        self.ids = df["ID"].astype(str).tolist()
        self.labels = df["label19"].astype(int).tolist()
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "label": self.labels[i] - 1}


class LLMCollator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labels = torch.tensor([f.pop("label") for f in feats], dtype=torch.long)
        batch = self.tok.pad(feats, padding=True, return_tensors="pt")
        batch["label"] = labels
        return batch


class LLMRegressor(nn.Module):
    """Causal trunk, last-token pooling, small ordinal head.

    Last-token pooling instead of mean pooling: with a causal mask only the last position
    has seen the whole sentence. The head runs in fp32 on a bf16 trunk, since an 18-way
    ordinal decision needs more precision than bf16 has near the cut points.
    """

    def __init__(self, name, objective, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 dropout=0.1, target_modules=None, grad_ckpt=False):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        trunk = AutoModel.from_pretrained(name, dtype=torch.bfloat16, trust_remote_code=True)
        trunk.config.use_cache = False
        # Sentences are short (p95 = 40 tokens), so activations are small next to the
        # weights. Checkpointing buys nothing here and costs an extra forward pass per step.
        if grad_ckpt and hasattr(trunk, "gradient_checkpointing_enable"):
            trunk.gradient_checkpointing_enable()
            trunk.enable_input_require_grads()
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"],
        )
        self.trunk = get_peft_model(trunk, cfg)
        h = trunk.config.hidden_size
        self.objective = objective
        out = 1 if objective == "reg" else (N_CLASSES - 1 if objective == "corn" else N_CLASSES)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, out)).float()

    def forward(self, input_ids, attention_mask, **_):
        hs = self.trunk(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1                       # right padding
        pooled = hs[torch.arange(hs.size(0), device=hs.device), last]
        return self.head(pooled.float())

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def predict_scores(model, loader, objective, device):
    """One score per row. No autocast needed, the trunk is already bf16."""
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("label", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            if objective == "reg":
                s = logits.squeeze(-1)
            elif objective == "corn":
                s = corn_scores(logits)
            else:
                s = expected_scores(logits)
            out.append(s.float().cpu().numpy())
    return np.concatenate(out)
