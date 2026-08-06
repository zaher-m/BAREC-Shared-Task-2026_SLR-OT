# artifacts/

What the pipeline produces. The score caches and logs are committed because everything
downstream, including `make verify` and every experiment, runs from them on CPU; the weights
are not.

| directory | tracked | contents |
|---|---|---|
| `member_scores/` | yes | per member: `scores_<tag>.npz` (dev/test scores and labels), `blind_scores_<tag>.npy`, `meta_<tag>.json`; plus `blind_ids.npy` |
| `logs/` | yes | training and campaign logs; `analysis/contamination.py` reads these |
| `models/` | no | per-member checkpoints (~51 GB); retrainable from `configs/campaigns/` |
| `ensembles/` | no | written by `slra_ot.cli.select_ensemble`, regenerable |
| `regenerated/` | no | scratch root; every script's default output, so re-runs never touch a committed record |

The uploaded files live in [submissions/](../submissions/), not here. Set `SLRA_OT_ARTIFACTS`
to move this root elsewhere.
