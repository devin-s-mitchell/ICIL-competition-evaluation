"""One demonstration -> the behavior prompt BPP expects. Pure numpy.

Reproduces PromptActionChunker on a single demo: observations at frames 0, chunk_n,
2*chunk_n, …; actions at full rate reshaped (P, chunk_n, A); mask all False (nothing
ignored). `pad_end` is BPP's `pad_end_prompt_actions`: with 'zeros' or 'repeat' a
trailing partial chunk is kept and padded, with 'no' it is dropped. A demo shorter than
one chunk always yields one zero-padded chunk. Each skill's chunk size and padding come
from `spec.json` (`skills.<skill>.environment`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rotations import actions_7_to_10, axis_angle_to_rotation_6d

PAD_MODES = ("no", "zeros", "repeat")


@dataclass
class PromptInfo:
    steps: int
    chunks: int
    padded_steps: int


def chunk_layout(steps: int, chunk_n: int, pad_end: str = "zeros") -> tuple[int, int]:
    """(chunks P, padded length) for a demo of `steps` actions."""
    if steps <= 0:
        raise ValueError("a demonstration needs at least one step")
    if pad_end not in PAD_MODES:
        raise ValueError(f"pad_end must be one of {PAD_MODES}")
    downsampled = steps // chunk_n
    less_than_one = downsampled == 0
    if less_than_one:
        downsampled = 1
    partial = steps % chunk_n != 0
    if partial and not less_than_one and pad_end in ("zeros", "repeat"):
        downsampled += 1
    return downsampled, downsampled * chunk_n


def chunk_actions(
    actions: np.ndarray, chunk_n: int, pad_end: str, max_chunks: int | None = None
) -> tuple[np.ndarray, np.ndarray, PromptInfo]:
    """(P, chunk_n, A) action chunks, the observation frame indices and the layout."""
    actions = np.asarray(actions, dtype=np.float32)
    steps = int(actions.shape[0])
    chunks, padded = chunk_layout(steps, chunk_n, pad_end)
    if max_chunks is not None and chunks > max_chunks:
        raise ValueError(f"prompt has {chunks} chunks; the model accepts at most {max_chunks}")
    idx = np.arange(0, padded, chunk_n)[:chunks]
    idx = np.minimum(idx, steps - 1)  # only matters for demos shorter than one chunk
    a = actions[:padded]
    if a.shape[0] < padded:
        pad_len = padded - a.shape[0]
        if pad_end == "repeat" and a.shape[0] > 0 and steps >= chunk_n:
            pad = np.repeat(a[-1:], pad_len, axis=0)
        else:
            pad = np.zeros((pad_len, a.shape[1]), np.float32)
        a = np.concatenate([a, pad], axis=0)
    return a.reshape(chunks, chunk_n, a.shape[1]), idx, PromptInfo(steps, chunks, padded)


def build_prompt(
    demo: dict[str, Any], chunk_n: int, max_chunks: int | None = None, pad_end: str = "zeros"
) -> tuple[dict[str, Any], PromptInfo]:
    """LIBERO: two cameras, end-effector pose (rot6d), gripper; 7-d actions widened to 10."""
    a10 = actions_7_to_10(np.asarray(demo["actions"], dtype=np.float32))
    chunks, idx, info = chunk_actions(a10, chunk_n, pad_end, max_chunks)
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
        "action": chunks,
        "mask": np.zeros((info.chunks,), dtype=bool),
    }
    return prompt, info


def build_draw_prompt(
    demo: dict[str, Any], chunk_n: int, max_chunks: int | None = None, pad_end: str = "no"
) -> tuple[dict[str, Any], PromptInfo]:
    """DrawAnything: one board image, pen position and pen state; 3-d actions (x, y, pen_down)."""
    chunks, idx, info = chunk_actions(
        np.asarray(demo["actions"], np.float32), chunk_n, pad_end, max_chunks
    )
    prompt = {
        "obs": {
            "image": np.asarray(demo["image"], dtype=np.uint8)[idx],
            "agent_pos": np.asarray(demo["agent_pos"], dtype=np.float32)[idx],
            "pen_down": np.asarray(demo["pen_down"], dtype=np.float32).reshape(-1, 1)[idx],
        },
        "action": chunks,
        "mask": np.zeros((info.chunks,), dtype=bool),
    }
    return prompt, info
