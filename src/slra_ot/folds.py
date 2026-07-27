"""Document-grouped folds over all the labelled data.

The Strict Track may only train on the BAREC train split. The Open Track allows any public
data, and BAREC dev and test are published with their gold labels, so all 69,441 labelled
sentences are usable here: 27% more than the 54,845 the strict system had.

The catch is that using dev and test for training costs the held-out set that early
stopping and the 18 thresholds need. So instead of one train/holdout cut: K grouped folds,
train K models per config, and stitch the held-out fold predictions back into out-of-fold
scores covering every row. Selection and calibration then run on those, and the blind
prediction averages the K models.

Folds are grouped by document, because sentences from the same document are very similar.
A random row split leaks a document across the boundary and makes everything look better
than it is.

Written once to data/processed/folds.parquet so different runs stay row-aligned.
"""
import numpy as np
import pandas as pd

from slra_ot.paths import PROCESSED, RAW_SPLITS, processed

FOLDS = PROCESSED / "folds.parquet"
LABELLED_SPLITS = ("train", "validation", "test")


def build(n_folds=5, seed=42):
    parts = []
    for split in LABELLED_SPLITS:
        df = pd.read_parquet(RAW_SPLITS[split])
        parts.append(pd.DataFrame({
            "ID": df["ID"].astype(str),
            "label19": df["Readability_Level_19"].astype(int),
            "document": df["Document"].astype(str),
            "orig_split": split,
            "word_count": df["Word_Count"].astype(int),
        }))
    pool = pd.concat(parts, ignore_index=True)

    # Balance on document size and mean level, so no fold ends up with all the long or all
    # the hard documents: sort by mean level, then hand each document to the lightest fold.
    docs = (pool.groupby("document")
                .agg(n=("ID", "size"), mean_level=("label19", "mean"))
                .reset_index())
    rng = np.random.default_rng(seed)
    docs = docs.iloc[rng.permutation(len(docs))].sort_values("mean_level", kind="stable")
    fold_load = np.zeros(n_folds)
    assign = {}
    for r in docs.itertuples():
        f = int(np.argmin(fold_load))
        assign[r.document] = f
        fold_load[f] += r.n
    pool["fold"] = pool["document"].map(assign).astype(int)

    pool.to_parquet(FOLDS, index=False)
    print(f"pool: {len(pool)} sentences, {pool.document.nunique()} documents -> {FOLDS}")
    print(pool.groupby("fold").agg(n=("ID", "size"), docs=("document", "nunique"),
                                   mean_level=("label19", "mean")).to_string())
    print("\nlabel distribution per fold (should be near-identical):")
    print(pd.crosstab(pool.label19, pool.fold, normalize="columns").round(3).to_string())
    return pool


def load_pool():
    return pd.read_parquet(FOLDS)


def fold_frames(variant, fold, n_folds=5):
    """(train, holdout) frames of [ID, text, label19] for one fold."""
    pool = load_pool()
    texts = pd.concat([pd.read_parquet(processed(variant, s)) for s in LABELLED_SPLITS],
                      ignore_index=True)
    texts["ID"] = texts["ID"].astype(str)
    df = pool.merge(texts[["ID", "text"]], on="ID", how="left", validate="one_to_one")
    assert df["text"].notna().all(), "missing preprocessed text for some IDs"
    tr = df[df.fold != fold][["ID", "text", "label19"]].reset_index(drop=True)
    ho = df[df.fold == fold][["ID", "text", "label19"]].reset_index(drop=True)
    return tr, ho
