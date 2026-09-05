# Operations

```bash
icilval keys generate --out keys                      # validator signing key (keep keys/ private)
icilval store init store --key keys/validator.ed25519 --pool-id <pool_id>
icilval daemon --store store --pool pools/2026.09-v1 --key keys/validator.ed25519 \
  --queue queue/queue.json --runs runs --admin-token "$ICIL_ADMIN_TOKEN" \
  --live https://<dashboard> --live-token "$ICIL_LIVE_TOKEN" \
  --mirror <owner>/icil-competition-results --docker-image icilval/model:dev
```

The daemon pops the queue, runs the duel (each side in the model container with `--network none`),
publishes the signed record, mirrors the store and posts live frames. `--once` runs a single entry.
Without `--docker-image` the sides run in-process (the BPP conda environment).

Smoke test end to end (in the BPP environment):

```bash
MUJOCO_GL=egl icilval smoke --store /tmp/store --pool pools/smoke --model-dir ~/.cache/icilval/models/bpp-libero-genesis --same-model
icilval store verify /tmp/store
```

Genesis: convert the public checkpoint, publish it as `spec.baseline.repo`, pin the revision,
then `icilval genesis --king <repo>@<revision> …` (or simply queue it on an empty throne).
