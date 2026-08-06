# Submissions

The complete upload record. `make verify` re-derives every file here from the committed score
caches and diffs it row by row; blind scores are the platform's and are marked † in
[docs/results.md](../docs/results.md).

- `submitted/` — the 8 uploads to CodaBench: the `prediction` CSV exactly as it went inside
  the uploaded archive, plus the `meta.json` (or `ensemble.json` for sub01) recording pool,
  weights, cuts and the CV estimate each was chosen on. Archives themselves are gitignored;
  `zip prediction.zip prediction` recreates the upload format.
- `candidates/` — built but never uploaded: sub05 (superseded the same evening) and the two
  post-deadline sub10 variants from [exp018](../experiments/exp018_peer_blend/), kept as the
  record of a tested-and-rejected idea.

