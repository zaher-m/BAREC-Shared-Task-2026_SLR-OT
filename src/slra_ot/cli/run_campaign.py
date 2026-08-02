"""Run a training campaign from configs/campaigns/*.yaml.

One job at a time. Members whose scores already exist are skipped, so an interrupted queue
can just be restarted. Each member's output goes to artifacts/logs/train_<tag>.log, and the
blind set is scored right after training so the member is ready to select on.

    python -m slra_ot.cli.run_campaign configs/campaigns/01_backbone_coverage.yaml --dry-run
    python -m slra_ot.cli.run_campaign configs/campaigns/04_clean_pool.yaml --stream a
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

from slra_ot.paths import CONFIGS, LOGS, MEMBER_SCORES, ROOT

TRAINERS = {"encoder": "train_encoder", "silver": "train_silver", "llm": "train_llm"}
META_KEYS = {"tag", "backbone", "stream"}


def flags(params):
    """{name: value} -> command-line flags. True means a bare flag, False/None is skipped."""
    out = []
    for k, v in params.items():
        if v is None or v is False:
            continue
        out.append(f"--{k}")
        if v is not True:
            out.append(str(v))
    return out


def member_command(cfg, member, backbones):
    params = {k: v for k, v in cfg.get("defaults", {}).items()}
    params.update({k: v for k, v in member.items() if k not in META_KEYS})
    return [sys.executable, "-m", f"slra_ot.cli.{TRAINERS[cfg['trainer']]}",
            "--model", backbones[member["backbone"]], "--tag", member["tag"]] + flags(params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a campaign yaml (see configs/campaigns/)")
    ap.add_argument("--stream", default=None, help="only members carrying this stream label")
    ap.add_argument("--dry-run", action="store_true", help="print the queue and exit")
    ap.add_argument("--force", action="store_true", help="retrain members that already exist")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    backbones = yaml.safe_load((CONFIGS / "backbones.yaml").read_text())
    members = [m for m in cfg["members"]
               if args.stream is None or m.get("stream") == args.stream]

    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"campaign {cfg['name']} | trainer {cfg['trainer']} | {len(members)} members"
          + (f" | stream {args.stream}" if args.stream else ""))

    for m in members:
        tag = m["tag"]
        if not args.force and (MEMBER_SCORES / f"scores_{tag}.npz").exists():
            print(f"skip {tag} (already trained)")
            continue
        cmd = member_command(cfg, m, backbones)
        log = LOGS / f"train_{tag}.log"
        if args.dry_run:
            print(f"  {' '.join(cmd)}  > {log}")
            continue
        print(f">>> {time.strftime('%H:%M:%S')} {tag}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        print(f"<<< {time.strftime('%H:%M:%S')} {tag} (exit {rc})", flush=True)
        if rc == 0 and cfg.get("score_blind", True):
            subprocess.call([sys.executable, "-m", "slra_ot.cli.score_split",
                             "--split", "blind", "--tags", tag], cwd=ROOT)

    print(f"=== campaign {cfg['name']} done ===")


if __name__ == "__main__":
    main()
