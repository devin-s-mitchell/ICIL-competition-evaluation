# Pools

A pool is a content-addressed directory the validator draws units from:

```
pool.json          manifest (tasks, variants, per-axis eligibility, pool_id = sha256 of the rest)
bddl/<group>/…     LIBERO BDDL files (originals, LIBERO-PRO swaps, table-swapped rewrites)
init/<group>/…     initial states as npz (converted from LIBERO's pickled .pruned_init at build time)
demos/<task>/…     demonstrations as npz (upright frames, proprioception, actions, initial state)
```

Build (needs the BPP conda environment and the raw downloads under `~/.cache/icilval/raw`):

```bash
MUJOCO_GL=egl icilval pools build --out pools/2026.09-v1 [--stage base spatial environment object composition finalize] [--limit N]
icilval pools verify pools/2026.09-v1
icilval pools push pools/2026.09-v1 --repo <owner>/icil-competition-pools
```

Validation during the build keeps, per variant, the initial states (or displacement slots) where
the scene is feasible and the goal is not already satisfied. The pool id is pinned in
`spec.json` (`pools.pool_id`); the validator refuses a pool that does not match.

Raw inputs: `yifengzhu-hf/LIBERO-datasets` (spatial/goal/object/10), `zhouxueyang/LIBERO-Pro`,
`austinpatel/libero_gen_goal_chain_hdf5` (first-step and selected chain views),
`austinpatel/libero_gen_spatial_combination_hdf5` (selected view), BPP's LIBERO checkout.

`icilval pools generate` runs BPP's LIBERO-Gen scripts with `affordance.yaml` to add
organizer-generated tasks under `generated/`; those are never published as training data.

## Pool 2026.09-v1

`pool_id` `7bf31989bda4b4147faf68d26f7102fc581e39c73ccd1101c467cc328847d259` — 72 tasks, 720 demonstrations.

| variant kind | count | eligible | valid slots (min / median / max) |
|---|---|---|---|
| level | 190 | 190 | 108 / 400 / 400 |
| lighting | 40 | 40 | 50 / 50 / 50 |
| pro_pose | 40 | 37 | 0 / 50 / 50 |
| pro_swap | 40 | 40 | 50 / 50 / 50 |
| table | 80 | 80 | 20 / 20 / 20 |

Eligible per axis: composition 19, environment 120, object 22, spatial 267.
