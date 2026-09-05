# icilval — ICIL competition validator

Python 3.10, package `icilval` under `src/`. Scores BPP-architecture checkpoints on four
perturbation axes (spatial, environment, object, composition), runs King-of-the-Hill duels,
publishes a signed append-only store, mirrors it to a Hugging Face dataset, and posts live
frames to the dashboard. Follows the conventions in `../CLAUDE.md`.

## Commands

- Host env (pure, no simulator): `uv venv .venv && uv pip install -e ".[dev]"`; `ruff check . && ruff format --check .`; `pytest -m "not sim and not gpu and not container"`.
- Simulator/GPU env: the pinned BPP conda env `/root/miniconda3/envs/bp` (`BP=/root/miniconda3/envs/bp/bin/python`); `$BP -m pytest -m sim`, `$BP -m pytest -m gpu`.
- End to end: `icilval smoke --store <dir>` then `icilval store verify <dir>`.
- Container: `docker/build.sh`; `pytest -m container`.

## Rules

- `spec.json` and `store-schema.json` are the contract. No number from them is duplicated as a literal; read through `icilval.spec`.
- Submissions are `model.safetensors` + `config.yaml` only. The validator never unpickles entrant data. Model-side runs happen in a container with `--network none`.
- Stored scores are fractions `[0, 1]`; `duel.score_margin` is percentage points; convert only in `icilval.duel.score`.
- Simulator imports (`libero`, `robosuite`, `torch`, `behavior_prompting`) are function-local so the host CLI imports without them.
- Pools are built offline (`icilval pools build`); pickled `.pruned_init`/hdf5 are read only at build time; runtime reads npz.
- Everything published is deterministic from `spec.json` + `pool_id` + the two model refs: unit lists, seeds, ids.
- BPP is vendored as a pinned git submodule at `vendor/behavior_prompting` (with `deps/LIBERO`); do not import its runner/workspace/dataset code, only the policy/model classes and LIBERO env utilities.

## Conventions

- Commit titles: `(feat): short description`, `(fix): …`; one coherent change per commit,
  brief bullets in the body when needed; commit at logical checkpoints.
- Keep PRs focused and reviewable; include tests and a CHANGELOG entry.
- Reports and evaluation results are plain files in the repository or run directory, not hosted
  artifacts.
