# Changelog

## Unreleased

### Skills (spec v2, store schema 2, live schema 2)

- (feat): the four perturbation axes become **skills**: `pick_and_place` (LIBERO, `bpp_libero_v1`) and `draw_anything` (DrawAnything-Sim, `bpp_draw_v1`). Each skill is one success rate; the final score is the mean over skills; a skill's units are spread evenly over its perturbation groups (spatial / environment / object; rotation), which are published per unit and never scored. Composition is dropped (BPP's separate chain domain).
- (feat): pick-and-place scope follows BPP's definition — one Grasp then one Place (`skills.pick_and_place.task_filter`): 27 base + 21 object-swap tasks survive from the v1 pool.
- (feat): DrawAnything-Sim skill: `sim/draw_env.py` (BPP's `DrawEnv` behind the LIBERO-shaped surface), `sim/draw_episode.py` (Chamfer-threshold success, BPP's idle stop), `model/draw.py`, `pools/build_draw.py` (the human-drawn evaluation set; organizer-generated drawings through BPP's generator), drawing instances derived from ids (no init files).
- (feat): submissions hold one directory per skill; `fingerprint.check_submission` reports per skill; `side_runner` loads each skill's checkpoint in turn; `convert-ckpt` renames `umi_day.` targets.
- (feat): `pools upgrade` migrates a schema-1 pool without re-simulation; pool `2026.09-v2` (`ae9645cd…`): 48 pick-and-place tasks, 50 drawings, 730 demonstrations.
- (feat): draw-board tests (`-m sim`) and a drawing parity test (`-m gpu`); the container image runs pygame headless.
- smoke (spec v2): `icilval smoke` on the two-skill genesis over the upgraded smoke pool publishes 2 signed events with 12 clips; both sides reproduce identical trajectories on both simulators (6 ties, Δ = 0, identical clip hashes, crown stays); `store verify` passes; the dashboard renders the store.
- parity (draw): the converted `austinpatel/drawanything_sim` checkpoint redraws human demonstrations on a turned board within 4 px on 0.79 of 80 calibration units (median best Chamfer 2.5 px).

- (feat): contract (`spec.json`, `store-schema.json`), ids, scoring, unit derivation, signed store, queue, admin intake, live reporter.
- (feat): BPP model path (converter, fingerprint, inference), simulator runner, pool builders, duel orchestration, mirror, daemon, Docker recipe.
- parity: converted `austinpatel/libero` checkpoint reaches 0.96 success on libero_spatial (10 tasks x 5 initial states, one prompt demo).
- smoke: `icilval smoke` (genesis + genesis-vs-genesis, size smoke, in-process) publishes 2 signed events with 16 clips; both sides reproduce identical trajectories (8 ties, Δ = 0, crown stays); `store verify` passes.
- container mode: `icilval duel --docker-image icilval/model:dev` runs both sides inside the image with `--network none` (8 units, 3 clips each, identical outcomes on both sides, `store verify` OK); `pytest -m container` passes 3/3.
- live path: `icilval smoke --live http://127.0.0.1:20202 --live-token …` delivered 22 frames to the dashboard's `/api/live` (all 200) and they stream back as `progress` events; a validator-built frame with per-axis progress and `recent_media` parses on the strict route.
- published: the genesis checkpoint as `robotensor/bpp-libero-genesis` and a smoke duel on pool `2026.09-v1` as the public dataset `robotensor/icil-competition-results`, which the dashboard reads.
