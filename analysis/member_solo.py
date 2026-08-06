"""The per-member solo table behind docs/members.md.

Solo QWK here is test-fitted: cuts fitted and scored on the same public test split, which is
in-sample and optimistic, quoted because it is how every member was compared during the
project. Protocol and recipe come from each member's metadata and the campaign configs.

    python analysis/member_solo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slra_ot.members import aligned, load_members  # noqa: E402
from slra_ot.metrics import fast_qwk  # noqa: E402
from slra_ot.thresholds import FastQWKThresholds, labels_from_thresholds  # noqa: E402

AD = ("_ad", "_ad2", "_ad3", "_ad4")


def main():
    ms = load_members(drop_degenerate=False)
    ids, X, y = aligned(ms, "test")
    rows = []
    for j, m in enumerate(ms):
        if float(X[:, j].std()) == 0.0:
            rows.append((0.0, m["tag"], "collapsed"))
            continue
        th = FastQWKThresholds(rounds=8).fit(X[:, j], y).thresholds_
        q = fast_qwk(y - 1, labels_from_thresholds(X[:, j], th) - 1) * 100
        proto = ("silver" if "slv" in m["tag"] else
                 "_ad" if m["tag"].endswith(AD) else "clean")
        rows.append((q, m["tag"], proto))
    rows.sort(reverse=True)
    print(f"{'solo QWK':>9s}  {'member':28s} protocol")
    for q, tag, proto in rows:
        print(f"{('collapsed' if proto == 'collapsed' else f'{q:9.2f}'):>9s}  {tag:28s} {proto}")


if __name__ == "__main__":
    main()
