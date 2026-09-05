# Changelog

## Unreleased

- (feat): contract (`spec.json`, `store-schema.json`), ids, scoring, unit derivation, signed store, queue, admin intake, live reporter.
- (feat): BPP model path (converter, fingerprint, inference), simulator runner, pool builders, duel orchestration, mirror, daemon, Docker recipe.
- parity: converted `austinpatel/libero` checkpoint reaches 0.96 success on libero_spatial (10 tasks x 5 initial states, one prompt demo).
- smoke: `icilval smoke` (genesis + genesis-vs-genesis, size smoke, in-process) publishes 2 signed events with 16 clips; both sides reproduce identical trajectories (8 ties, Δ = 0, crown stays); `store verify` passes.
- container mode: `icilval duel --docker-image icilval/model:dev` runs both sides inside the image with `--network none` (8 units, 3 clips each, identical outcomes on both sides, `store verify` OK); `pytest -m container` passes 3/3.
- live path: `icilval smoke --live http://127.0.0.1:20202 --live-token …` delivered 22 frames to the dashboard's `/api/live` (all 200) and they stream back as `progress` events; a validator-built frame with per-axis progress and `recent_media` parses on the strict route.
