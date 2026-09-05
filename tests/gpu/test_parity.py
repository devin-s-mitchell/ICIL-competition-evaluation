"""The converted genesis checkpoint must reproduce BPP's LIBERO-spatial performance."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from icilval.model.bpp import BPPPolicy
from icilval.model.fingerprint import check_submission
from icilval.sim.episode import run_episode
from icilval.sim.libero_env import LiberoEnv, load_init_states
from icilval.spec import _repo_root

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

N_TASKS = int(os.environ.get("ICILVAL_PARITY_TASKS", "10"))
N_INIT = int(os.environ.get("ICILVAL_PARITY_INITS", "5"))
FLOOR = float(os.environ.get("ICILVAL_PARITY_FLOOR", "0.90"))


def test_genesis_passes_fingerprint(spec, genesis_dir):
    rep = check_submission(genesis_dir, spec, (_repo_root() or Path.cwd()) / "arch")
    assert rep.ok, rep.errors


def test_parity_libero_spatial(spec, genesis_dir, smoke_pool_or_skip):
    pool = smoke_pool_or_skip
    tasks = [t for t in pool.tasks.values() if t.suite == "libero_spatial"][:N_TASKS]
    if not tasks:
        pytest.skip("pool has no libero_spatial tasks")
    policy = BPPPolicy(genesis_dir, (_repo_root() or Path.cwd()) / "arch", spec)
    policy.load()
    from icilval.pools.demos import load_demo

    successes, n = 0, 0
    for task in tasks:
        env = LiberoEnv(pool.path(task.bddl), spec)
        states = load_init_states(pool.path(task.init))
        demo = load_demo(pool.path("demos") / f"{task.demos[0]}.npz")
        for i in range(min(N_INIT, len(states))):
            unit = {
                "unit_id": f"sp-{i:03d}",
                "axis": "spatial",
                "seed": 1000 + i,
                "max_steps": task.max_steps,
                "perturbation": {"kind": "none"},
            }
            res = run_episode(env, policy, unit, np.asarray(states[i]), demo, spec)
            assert not res.void, res.error
            successes += int(res.success)
            n += 1
        env.close()
    assert successes / n >= FLOOR, f"success {successes}/{n}"


@pytest.fixture(scope="session")
def smoke_pool_or_skip():
    root = Path(
        os.environ.get("ICILVAL_SMOKE_POOL", Path.home() / ".cache" / "icilval" / "pools" / "smoke")
    )
    if not (root / "pool.json").exists():
        pytest.skip(f"no pool at {root}")
    from icilval.pools.schema import Pool

    return Pool.load(root)
