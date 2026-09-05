"""One demonstration -> the behavior prompt BPP expects. Pure numpy; images stay 128 px here.

Reproduces PromptActionChunker(chunk_n=20, pad_end='zeros') on a single demo:
observations at frames 0, 20, 40, …; actions at full rate reshaped (P, 20, 10)
with the trailing partial chunk zero-padded; mask all False (nothing ignored).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rotations import actions_7_to_10, axis_angle_to_rotation_6d


@dataclass
class PromptInfo:
    steps: int
    chunks: int
    padded_steps: int


def chunk_layout(steps: int, chunk_n: int) -> tuple[int, int]:
    """(chunks P, padded length) for a demo of `steps` actions, pad_end='zeros'."""
    if steps <= 0:
        raise ValueError("a demonstration needs at least one step")
    downsampled = steps // chunk_n
    less_than_one = downsampled == 0
    if less_than_one:
        downsampled = 1
    partial = steps % chunk_n != 0
    if partial and not less_than_one:
        downsampled += 1
    return downsampled, downsampled * chunk_n


def build_prompt(
    demo: dict[str, Any], chunk_n: int, max_chunks: int | None = None
) -> tuple[dict[str, Any], PromptInfo]:
    actions = np.asarray(demo["actions"], dtype=np.float32)
    steps = int(actions.shape[0])
    chunks, padded = chunk_layout(steps, chunk_n)
    if max_chunks is not None and chunks > max_chunks:
        raise ValueError(f"prompt has {chunks} chunks; the model accepts at most {max_chunks}")
    idx = np.arange(0, padded, chunk_n)[:chunks]
    idx = np.minimum(idx, steps - 1)  # only matters for demos shorter than one chunk
    a10 = actions_7_to_10(actions[:padded])
    if a10.shape[0] < padded:
        a10 = np.concatenate([a10, np.zeros((padded - a10.shape[0], 10), np.float32)], axis=0)
    prompt = {
        "obs": {
            "agentview_rgb": np.asarray(demo["agentview"], dtype=np.uint8)[idx],
            "eye_in_hand_rgb": np.asarray(demo["eye_in_hand"], dtype=np.uint8)[idx],
            "ee_pos": np.asarray(demo["ee_pos"], dtype=np.float32)[idx],
            "ee_ori": axis_angle_to_rotation_6d(
                np.asarray(demo["ee_ori"], dtype=np.float64)[idx]
            ).astype(np.float32),
            "gripper_states": np.asarray(demo["gripper"], dtype=np.float32)[idx],
        },
        "action": a10.reshape(chunks, chunk_n, 10),
        "mask": np.zeros((chunks,), dtype=bool),
    }
    return prompt, PromptInfo(steps=steps, chunks=chunks, padded_steps=padded)
