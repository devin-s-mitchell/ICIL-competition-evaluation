# icilval

Validator for the RoboTensor **one-demonstration in-context imitation learning** competition.
A Behavior Prompting Policy (BPP) checkpoint is shown one demonstration of a task, no language,
and is scored on four axes of change: spatial, environment, object, composition. The final score
is the mean of the four; a challenger takes the crown by beating the reigning model's average by
`duel.score_margin` points on an identical unit list. Everything published is signed and
reproducible from `spec.json`, the pool id and the two model references.

- `spec.json`, `store-schema.json` — the contract (also vendored by the dashboard).
- `arch/` — the allow-listed architecture template exported from the genesis checkpoint.
- `affordance.yaml` — object → operator table driving held-out task generation.
- `docs/` — protocol, submissions, pools, operations.
- `vendor/behavior_prompting` — BPP pinned as a git submodule (`git submodule update --init --recursive`).

## Quick start (organizer)

```bash
uv venv --python 3.10 .venv && uv pip install -e ".[dev]"          # host tools, no simulator
pytest -m "not sim and not gpu and not container"                   # pure tests

# simulator work happens in the BPP conda environment (see docs/pools.md)
MUJOCO_GL=egl icilval pools build --out pools/2026.09-v1
MUJOCO_GL=egl icilval convert-ckpt --ckpt libero_behavior_prompting.ckpt --out models/genesis --emit-arch arch
MUJOCO_GL=egl icilval smoke --store /tmp/store --pool pools/smoke --model-dir models/genesis --same-model
icilval store verify /tmp/store
```

## Layout

```
src/icilval/
  spec.py canon.py ids.py rng.py        contract, canonical JSON + ed25519, ids, hash RNG
  model/   rotations prompt fingerprint convert bpp
  sim/     bddl libero_env perturb lighting episode video
  pools/   schema sources validate build build_gen demos units hub
  duel/    score side_runner orchestrate
  store/   records writer verify mirror
  queue.py admin.py live.py daemon.py submission.py cli.py
```

Licensed under Apache-2.0. BPP and LIBERO are MIT; LIBERO-PRO data is CC-BY-4.0.
