# Pools

A pool is a content-addressed directory the validator draws units from:

```
pool.json          manifest (tasks, variants, per-skill eligibility, pool_id = sha256 of the rest)
bddl/<group>/…     LIBERO BDDL files (originals, LIBERO-PRO swaps, table-swapped rewrites)
init/<group>/…     LIBERO initial states as npz (converted from pickled .pruned_init at build time)
demos/<task>/…     demonstrations as npz: LIBERO cameras + proprioception + actions + initial state,
                   or the drawing board's frames + pen state + actions + the strokes drawn
```

Build (needs the BPP conda environment and the raw downloads under `~/.cache/icilval/raw`):

```bash
MUJOCO_GL=egl icilval pools build --out pools/2026.09-v2 [--stage base spatial environment object draw finalize] [--limit N]
icilval pools verify pools/2026.09-v2
icilval pools push pools/2026.09-v2 --repo <owner>/icil-competition-pools
```

The LIBERO stages keep only pick-and-place tasks (`skills.pick_and_place.task_filter`: a single
On/In goal, no open / close / turn-on / push stage). Validation during the build keeps, per
variant, the initial states (or displacement slots) where the scene is feasible and the goal is
not already satisfied. The `draw` stage imports BPP's human-drawn evaluation set
(`eval_handmade.zarr`, 50 drawings, 5 demonstrations each); a drawing task has no initial-state
file — its `init_states_per_task` instances are board angles and cursor starts derived from the
task id. The pool id is pinned in `spec.json` (`pools.pool_id`); the validator refuses a pool that
does not match.

Raw inputs: `yifengzhu-hf/LIBERO-datasets` (spatial/goal/object/10), `zhouxueyang/LIBERO-Pro`,
`austinpatel/libero_gen_goal_chain_hdf5` (first-step view), `austinpatel/libero_gen_spatial_combination_hdf5`
(selected view), `austinpatel/drawanything_sim` (`eval_handmade.zarr.zip`, unpacked), BPP's checkout.

Organizer-generated tasks, never published as training data:

- `icilval pools generate` runs BPP's LIBERO-Gen scripts with `affordance.yaml` and imports the
  object-swap tasks that pass the pick-and-place filter under `generated/`.
- `icilval pools generate-draw --base-seed <secret>` runs BPP's `procedural_generate_drawings.py`
  and imports its drawings under `drawanything_generated/`.

Upgrading a schema-1 pool (four axes) keeps every pick-and-place task and its validated
variants without re-simulating, drops the rest, then runs the draw stage:

```bash
icilval pools upgrade --old pools/2026.09-v1 --out pools/2026.09-v2
```

## Pool 2026.09-v2

`pool_id` `ae9645cdbc2ed24cdbea436d6677925070f0a4221c604a09e18c96dbd0dad83a` — 98 tasks, 730
demonstrations; upgraded from `2026.09-v1` (`7bf31989…`) plus the drawing stage.

| skill | tasks | variants | eligible per perturbation group |
|---|---|---|---|
| pick_and_place | 27 base (libero_spatial 10, libero_object 10, libero_goal 6, libero_10 1) + 21 object-swap (LIBERO-Gen) | level 135, pro_swap 27, pro_pose 27, table 50, lighting 27 | spatial 188, environment 77, object 21 |
| draw_anything | 50 human drawings (eval_handmade) | — | rotation 50 |

Dropped from v1 as not pick-and-place: the drawer / stove / push tasks of libero_goal and their
LIBERO-Gen first-step twin, and every chain (libero_10 two-step tasks, LIBERO-Gen goal chains).

## Calibrating the drawing threshold

`skills.draw_anything.success.threshold` was chosen by running the converted BPP drawing
checkpoint over 80 units derived from this pool (eight `heavy` draws' worth of drawing units; the
per-unit best Chamfer distances are the `*_metric` values a duel publishes). The distribution of
the best Chamfer distance, in canvas pixels on a 12 px pen:

| percentile | 10 | 25 | 50 | 75 | 90 |
|---|---|---|---|---|---|
| best Chamfer (px) | 0.96 | 1.39 | 2.46 | 3.83 | 7.21 |

| threshold (px) | 2.5 | 3 | 3.5 | 4 | 5 | 6 | 12 |
|---|---|---|---|---|---|---|---|
| genesis success | 0.51 | 0.61 | 0.69 | 0.79 | 0.83 | 0.86 | 0.93 |

The threshold is 4 px — a third of the pen width, so a success is a faithful reproduction rather
than a rough one — where the baseline misses the long multi-part drawings (`long5_robot`,
`long3_horizon`, `f2`, `custom6`) and little else. Every episode finished within 400 steps; none
was void.
