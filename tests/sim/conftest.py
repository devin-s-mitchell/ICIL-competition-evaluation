from __future__ import annotations

import os
from pathlib import Path

import pytest

SMOKE_POOL = Path(
    os.environ.get("ICILVAL_SMOKE_POOL", Path.home() / ".cache" / "icilval" / "pools" / "smoke")
)


def _has_sim() -> bool:
    try:
        import libero  # noqa: F401
        import robosuite  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="session", autouse=True)
def _sim_env():
    if not _has_sim():
        pytest.skip("simulator not importable (run in the BPP conda env)")
    os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture(scope="session")
def smoke_pool():
    if not (SMOKE_POOL / "pool.json").exists():
        pytest.skip(f"no smoke pool at {SMOKE_POOL} (icilval pools build --limit 1 --out …)")
    from icilval.pools.schema import Pool

    return Pool.load(SMOKE_POOL)
