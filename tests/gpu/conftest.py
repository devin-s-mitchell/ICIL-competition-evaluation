from __future__ import annotations

import os
from pathlib import Path

import pytest

MODEL_DIR = Path(
    os.environ.get(
        "ICILVAL_GENESIS_DIR", Path.home() / ".cache" / "icilval" / "models" / "bpp-libero-genesis"
    )
)


@pytest.fixture(scope="session", autouse=True)
def _gpu_env():
    if os.environ.get("ICILVAL_TEST_GPU") != "1":
        pytest.skip("set ICILVAL_TEST_GPU=1 to run GPU tests")
    os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture(scope="session")
def genesis_dir():
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip(f"no converted genesis model at {MODEL_DIR}")
    return MODEL_DIR
