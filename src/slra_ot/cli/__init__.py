"""Pipeline stages. Run them from the repo root:

    python -m slra_ot.cli.preprocess --variant d3tok
    python -m slra_ot.cli.train_encoder --model <hf-id> --tag <tag> --objective emd
    python -m slra_ot.cli.score_split --split blind --all
    python -m slra_ot.cli.build_submission --name sub07_clean --clean-only
"""
