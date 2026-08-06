# Configs

- `backbones.yaml` — short name to Hugging Face id for the eight encoder backbones and the
  7B decoder.
- `campaigns/*.yaml` — the member-training queues, in the order they ran. Each entry is one
  member: backbone, objective, variant, seed, protocol flags. `slra_ot.cli.run_campaign`
  executes them and skips members whose score caches already exist, so an interrupted queue
  restarts cleanly. The 52 members that predate the campaign runner are inventoried in
  [docs/members.md](../docs/members.md) instead.
