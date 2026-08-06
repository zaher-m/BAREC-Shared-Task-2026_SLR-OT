# Analysis

Diagnostics that look at a system rather than test a hypothesis. Anything that decided a design
choice lives in [experiments/](../experiments/) instead.

| script | question |
|---|---|
| `error_analysis.py` | which confusions carry the QWK loss, and is the blind marginal shifted? |
| `label_shift_robustness.py` | are QWK-optimal cuts or distribution-matched cuts safer under simulated label shift? |
| `submission_decision.py` | given today's pool, does the best honest estimate clear the bar for spending a submission? |

The first two need a saved ensemble (`python -m slra_ot.cli.select_ensemble --name v1`). The
third reads the member pool directly.
