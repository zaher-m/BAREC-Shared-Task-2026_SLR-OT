"""Loading member scores and lining them up into a matrix.

Each trained member writes scores_<tag>.npz (dev/test scores on the 1..19 scale) and
blind_scores_<tag>.npy. Loading them here means the ensemble work is plain arithmetic.

Getting the document of a sentence: dev and test have a Document column in the parquets.
The blind set has no metadata, but its IDs look like BAREC-ST-2026_<doc>_<sent>, so the
document and the sentence order inside it can be read off the ID.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from slra_ot.paths import MEMBER_SCORES, RAW_SPLITS

# This one diverged: it predicts a constant (score sd = 0.000, solo QWK 41.0). A weighted
# mean survives it, but it breaks per-member standardisation, so drop it.
DEGENERATE = {"arabertv2_large_reg_ad"}


def load_members(tags=None, exclude=None, pred_dir=MEMBER_SCORES, drop_degenerate=True):
    """Return [{tag, npz, meta}] for every scores_*.npz found (or just `tags`)."""
    out = []
    for f in sorted(pred_dir.glob("scores_*.npz")):
        tag = f.name[len("scores_"):-len(".npz")]
        if tags is not None and tag not in tags:
            continue
        if drop_degenerate and tag in DEGENERATE:
            continue
        if exclude and any(s in tag for s in exclude):
            continue
        meta_p = pred_dir / f"meta_{tag}.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        out.append({"tag": tag, "npz": np.load(f, allow_pickle=True), "meta": meta})
    if tags is not None:
        order = {t: i for i, t in enumerate(tags)}
        out.sort(key=lambda m: order[m["tag"]])
    return out


def pinned_pool(where):
    """Read a member list from a pool file, or from pool.txt inside a directory.

    The pool grew from 39 to 76 members while the project ran, so an experiment that just
    globs whatever is on disk gives a different answer every month. Each experiment directory
    keeps the list it actually used, and that list is what reproduces its results.json.
    """
    p = Path(where)
    if p.is_dir():
        p = p / "pool.txt"
    tags = [ln.split("#")[0].strip() for ln in p.read_text().splitlines()]
    tags = [t for t in tags if t]
    missing = [t for t in tags if not (MEMBER_SCORES / f"scores_{t}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of the {len(tags)} members pinned in {p} have no cached scores "
            f"(first: {missing[0]}). Train them with slra_ot.cli.run_campaign, or edit "
            f"pool.txt to the pool you do have and expect different numbers."
        )
    return tags


def pinned_members(exp_dir):
    """load_members over exactly the pool recorded in <exp_dir>/pool.txt.

    Nothing is filtered out here: pool.txt is the whole story, including the collapsed member
    for the experiments that ran before it was found.
    """
    return load_members(tags=pinned_pool(exp_dir), drop_degenerate=False)


def aligned(members, split):
    """Line members up on the first member's ID order.

    Returns (ids, X, y), X of shape [N, M] in the order members was given.
    """
    ref = [str(x) for x in members[0]["npz"][f"{split}_ids"]]
    y = members[0]["npz"][f"{split}_labels"].astype(int)
    cols = []
    for m in members:
        ids = [str(x) for x in m["npz"][f"{split}_ids"]]
        s = m["npz"][f"{split}_scores"]
        idx = {i: k for k, i in enumerate(ids)}
        cols.append(np.asarray([s[idx[i]] for i in ref], dtype=float))
    return ref, np.vstack(cols).T, y


def blind_ids():
    return [str(x) for x in np.load(MEMBER_SCORES / "blind_ids.npy", allow_pickle=True)]


def blind_matrix(tags):
    """[N_blind, M] matrix of cached blind scores, in tag order."""
    return np.vstack([np.load(MEMBER_SCORES / f"blind_scores_{t}.npy") for t in tags]).T


# ---------------------------------------------------------------- documents

_BLIND_RE = re.compile(r"^(?P<doc>.+)_(?P<sent>\d+)$")
_SPLIT_ALIAS = {"dev": "validation"}


def _meta_frame(split):
    return pd.read_parquet(RAW_SPLITS[_SPLIT_ALIAS.get(split, split)])


def doc_structure(ids, split):
    """Return (document id, position in document) for each sentence ID.

    Position is the 0-based rank inside the document: corpus row order for dev/test, and
    the number at the end of the ID for blind.
    """
    ids = [str(i) for i in ids]
    if split == "blind":
        docs, order = [], []
        for i in ids:
            m = _BLIND_RE.match(i)
            if not m:                                  # unexpected id, treat as its own doc
                docs.append(i); order.append(0); continue
            docs.append(m.group("doc")); order.append(int(m.group("sent")))
        docs = np.asarray(docs)
        order = np.asarray(order, dtype=float)
    else:
        df = _meta_frame(split)
        dmap = dict(zip(df["ID"].astype(str), df["Document"].astype(str)))
        # corpus row order is the document's reading order
        omap = {str(r.ID): k for k, r in enumerate(df.itertuples())}
        docs = np.asarray([dmap[i] for i in ids])
        order = np.asarray([omap[i] for i in ids], dtype=float)

    pos = np.zeros(len(ids), dtype=int)
    for d in np.unique(docs):
        sel = np.where(docs == d)[0]
        pos[sel[np.argsort(order[sel])]] = np.arange(len(sel))
    return docs, pos


def word_counts(ids, split):
    """Word count per sentence. Available for blind too, since it comes from the text."""
    ids = [str(i) for i in ids]
    if split == "blind":
        df = pd.read_parquet(RAW_SPLITS["blind"])
        m = dict(zip(df["Sentence ID"].astype(str),
                     df["Sentence"].astype(str).str.split().str.len()))
    else:
        df = _meta_frame(split)
        m = dict(zip(df["ID"].astype(str), df["Word_Count"].astype(int)))
    return np.asarray([m[i] for i in ids], dtype=float)
