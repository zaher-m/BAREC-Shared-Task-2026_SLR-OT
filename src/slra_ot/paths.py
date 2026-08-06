"""All file locations used by the project, kept in one place.

Everything is relative to the repo root, so no path is tied to a particular machine.
Set SLRA_OT_DATA or SLRA_OT_ARTIFACTS to move the two big directories elsewhere.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = Path(os.environ.get("SLRA_OT_DATA", ROOT / "data"))
RAW = DATA / "raw"                  # BAREC parquets as published
PROCESSED = DATA / "processed"      # preprocessed text, folds, silver corpus
EXTERNAL = DATA / "external"        # third-party corpora (BAREC-10M)

BAREC10M = EXTERNAL / "barec10m"
BAREC10M_ANNOTATIONS = BAREC10M / "mr" / "Morphology_and_Readability"

ARTIFACTS = Path(os.environ.get("SLRA_OT_ARTIFACTS", ROOT / "artifacts"))
MODELS = ARTIFACTS / "models"               # per-member checkpoints
MEMBER_SCORES = ARTIFACTS / "member_scores"  # cached member scores + metadata
ENSEMBLES = ARTIFACTS / "ensembles"          # selected member sets + their thresholds
LOGS = ARTIFACTS / "logs"                    # training / campaign logs

# Scratch output for re-runs. Gitignored, so re-running an experiment or a build can never
# overwrite a committed result or prediction. Anything a script writes by default belongs
# here; the committed originals only change deliberately.
REGENERATED = ARTIFACTS / "regenerated"

SUBMISSIONS = ROOT / "submissions"
SUBMITTED = SUBMISSIONS / "submitted"        # every file uploaded, as uploaded
CANDIDATES = SUBMISSIONS / "candidates"      # built but never uploaded

CONFIGS = ROOT / "configs"
EXPERIMENTS = ROOT / "experiments"

# split name -> its parquet. blind has no labels.
RAW_SPLITS = {
    "train": RAW / "barec_sent_train.parquet",
    "validation": RAW / "barec_sent_validation.parquet",
    "test": RAW / "barec_sent_test.parquet",
    "blind": RAW / "blind_sent.parquet",
}


def submission_dir(name):
    """Where a named submission lives in the committed record, uploaded or not."""
    for root in (SUBMITTED, CANDIDATES):
        if (root / name).is_dir():
            return root / name
    return None


def processed(variant, split):
    """Cached preprocessed text for one (variant, split)."""
    return PROCESSED / f"proc_{variant}_{split}.parquet"


def silver(variant):
    return PROCESSED / f"silver_{variant}.parquet"
