"""The model-side image: builds, imports the simulator and torch, runs the CLI, and has no network."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.container
IMAGE = os.environ.get("ICILVAL_IMAGE", "icilval/model:dev")


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check, timeout=600
    )


@pytest.fixture(scope="session", autouse=True)
def _have_image():
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    if _docker("image", "inspect", IMAGE, check=False).returncode != 0:
        pytest.skip(f"image {IMAGE} not built (docker/build.sh)")


def test_cli_and_imports():
    out = _docker("run", "--rm", "--network", "none", IMAGE, "icilval", "--help").stdout
    assert "run-side" in out
    code = "import torch, libero, robosuite, behavior_prompting, icilval, nacl, safetensors, imageio_ffmpeg; print('ok')"
    out = _docker("run", "--rm", "--network", "none", IMAGE, "python", "-c", code).stdout
    assert "ok" in out


def test_no_network_inside():
    res = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        IMAGE,
        "python",
        "-c",
        "import urllib.request\ntry:\n    urllib.request.urlopen('https://huggingface.co', timeout=5); print('reachable')\nexcept Exception as e:\n    print('blocked')",
        check=False,
    )
    assert "blocked" in res.stdout


def test_clip_weights_cached():
    code = "import timm; timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, global_pool='', num_classes=0); print('cached')"
    res = _docker("run", "--rm", "--network", "none", IMAGE, "python", "-c", code, check=False)
    assert "cached" in res.stdout, res.stderr[-800:]
