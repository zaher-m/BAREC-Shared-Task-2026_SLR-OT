"""Write the document-grouped fold split over all 69,441 labelled sentences.

    python -m slra_ot.cli.build_folds --folds 5 --seed 42
"""
import argparse

from slra_ot.folds import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.folds, args.seed)


if __name__ == "__main__":
    main()
