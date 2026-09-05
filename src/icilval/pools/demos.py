"""Demonstrations: hdf5 (LIBERO / LIBERO-Gen) -> one npz per demo, and back into prompts and clips.

Stored per demo (`demos/<task_id>/<demo>.npz`):
  agentview   (T,128,128,3) uint8, upright (hdf5 stores the OpenGL orientation; we flip once here)
  eye_in_hand (T,128,128,3) uint8, upright
  ee_pos (T,3) ee_ori (T,3 axis-angle) gripper (T,2) joints (T,7) actions (T,7) float32
  init_state (D,) float64   the simulator state the demo started from
  meta: source file, demo key, index in file, image_convention="upright"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..canon import sha256_file


@dataclass
class DemoMeta:
    demo_id: str
    path: str
    steps: int
    source: str
    source_key: str
    source_index: int
    init_sha256: str


def select_indices(n_available: int, k: int) -> list[int]:
    """k evenly spread indices over [0, n). Deterministic, keeps the first demo."""
    if k >= n_available:
        return list(range(n_available))
    return sorted({int(round(i * (n_available - 1) / max(1, k - 1))) for i in range(k)})


def _sorted_demo_keys(keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k)


def import_hdf5(
    src: str | Path, out_dir: str | Path, task_id: str, k: int, *, demo_prefix: str = "demo"
) -> list[DemoMeta]:
    import h5py

    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas: list[DemoMeta] = []
    with h5py.File(src, "r") as h:
        data = h["data"]
        keys = _sorted_demo_keys(list(data.keys()))
        for n, idx in enumerate(select_indices(len(keys), k)):
            key = keys[idx]
            g = data[key]
            obs = g["obs"]
            demo_id = f"{task_id}/{demo_prefix}_{n:02d}"
            path = out_dir / f"{demo_prefix}_{n:02d}.npz"
            init_state = (
                np.asarray(g.attrs["init_state"], dtype=np.float64)
                if "init_state" in g.attrs
                else np.asarray(g["states"][0], dtype=np.float64)
            )
            arrays = {
                "agentview": np.asarray(obs["agentview_rgb"], dtype=np.uint8)[:, ::-1].copy(),
                "eye_in_hand": np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)[:, ::-1].copy(),
                "ee_pos": np.asarray(obs["ee_pos"], dtype=np.float32),
                "ee_ori": np.asarray(obs["ee_ori"], dtype=np.float32),
                "gripper": np.asarray(obs["gripper_states"], dtype=np.float32),
                "joints": np.asarray(obs["joint_states"], dtype=np.float32),
                "actions": np.asarray(g["actions"], dtype=np.float32),
                "init_state": init_state,
            }
            meta = {
                "demo_id": demo_id,
                "source": src.name,
                "source_key": key,
                "source_index": idx,
                "image_convention": "upright",
                "steps": int(arrays["actions"].shape[0]),
            }
            np.savez_compressed(path, meta=json.dumps(meta), **arrays)
            metas.append(
                DemoMeta(
                    demo_id=demo_id,
                    path=str(path.relative_to(out_dir.parent.parent))
                    if out_dir.parent.parent in path.parents
                    else str(path),
                    steps=meta["steps"],
                    source=src.name,
                    source_key=key,
                    source_index=idx,
                    init_sha256=sha256_file(path) if False else "",
                )
            )
    return metas


def load_demo(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def demo_frames(demo: dict[str, Any], upscale_factor: int = 2, stride: int = 1) -> list[np.ndarray]:
    from ..sim.video import side_by_side, upscale

    frames = []
    for t in range(0, demo["agentview"].shape[0], stride):
        a = upscale(demo["agentview"][t], upscale_factor)
        b = upscale(demo["eye_in_hand"][t], upscale_factor)
        frames.append(side_by_side(a, b))
    return frames


def render_demo(
    path: str | Path,
    out_mp4: str | Path,
    fps: int,
    video_cfg: dict[str, Any],
    upscale_factor: int = 2,
) -> str:
    from ..sim.video import encode_frames

    demo = load_demo(path)
    return encode_frames(demo_frames(demo, upscale_factor), out_mp4, fps, video_cfg)
