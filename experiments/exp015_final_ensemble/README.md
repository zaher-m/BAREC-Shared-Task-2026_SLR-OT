# E15. Final ensemble, and the saturation wall

**When** Aug 3 · **Verdict** a second silver member adds nothing; the pool is saturated

## Silver members are individually good

| silver member | solo test QWK | fresh twin | gain |
|---|---|---|---|
| `arabertv2_emd_slv` | 85.93 | `arabertv2_emd_c2` 84.93 | +1.00 |
| `arabertv2_soft_slv` | 85.78 | `arabertv2_soft_c2` 85.37 | +0.41 |

Silver pretraining reliably improves the member it is applied to. That part replicated.

## But the pool is saturated

| config | 1 silver member | 2 silver members | dQWK |
|---|---|---|---|
| cap30, k=0.35 | 86.713 | 86.632 | -0.08 |
| cap30, k=0.50 | 86.668 | 86.568 | -0.10 |
| cap30, k=0.65 | 86.703 | 86.599 | -0.10 |
| clean, k=0.35 | 86.624 | 86.634 | +0.01 |
| clean, k=0.50 | 86.598 | 86.575 | -0.02 |
| clean, k=0.65 | 86.635 | 86.542 | -0.09 |

Adding a second good member made the ensemble slightly worse in five of six configurations.
Individual magnitudes are near CV noise, but the sign is consistent, and it matches every
other pool-size result in this project:

- exp002: pool 10 -> 39 was worth +0.19
- exp004/exp010: pool 39 -> 66 was worth +0.007 (and negative before the collapsed member
  was removed)
- here: 70 -> 71 is -0.08

The curve flattened long ago and has now turned over. With ~70 highly-correlated members on
7,286 calibration rows, each additional column adds more selection variance than signal.

## Consequence for the plan

Adding members is finished as a strategy. The remaining silver queue (AraELECTRA-corn, an emd
reseed) was cancelled: more members at ~85.8 cannot help a pool whose best is 86.07 and whose
combination is already saturated.

What could still move the number is a substantially stronger member, one that would take large
weight and pull the combination up rather than dilute it. Hence `arabertv2_large_emd_slv`:
bert-large-arabertv2 with the same silver->gold recipe. The large backbone has more capacity to
exploit 439K extra sentences than the base one did, and its plain variant
(`arabertv2_large_corn`, 85.46) is already among the stronger clean members.

Gold LR is set to 1.2e-5 rather than 2e-5 for two reasons: large backbones want a lower rate,
and every silver member so far peaks at gold epoch 0 and then declines, so the model arrives
already converged on the task and 2e-5 immediately overfits.

## Fallback

If the large member does not land meaningfully above 86.07, the best honestly-estimated system
remains cap30 / k=0.35 / 1 silver member, CV 86.713 and implied blind 85.29. That is a marginal
improvement on the 85.2 that sub02 and sub04 both scored, and short of thylinao's 85.3.

Later note: the -0.08 read as saturation here was CV noise (exp016), and the capped pool it
recommends was the wrong family to compare in at all (exp017).

## Files

| File | Role | Writes |
|---|---|---|
| `pool.txt` | the exact member tags this experiment ran on | - |
| `results.json` | the recorded output, as produced | - |
| `run.py` | exp015: final selection, now with the silver members in the pool | scratch root |
