"""Runtime perturbations applied after the initial state is restored.

- displace_objects: LIBERO-PRO's initial-pose case and our level ladder (delta xy + yaw).
- apply_lighting / restore_lighting: the environment perturbation's light draw onto the MuJoCo model.
Both record exactly what they did so the verdict can publish it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..model.rotations import quat_multiply_wxyz, yaw_quaternion_wxyz
from .libero_env import LiberoEnv
from .perturb_math import direction_delta  # noqa: F401 - re-exported for callers


class Infeasible(Exception):
    """The perturbation left the scene in a state that must not be scored."""


def table_bounds(env: LiberoEnv, margin_m: float = 0.04) -> tuple[np.ndarray, np.ndarray] | None:
    """(xy_min, xy_max) of the nominal table top in world coordinates, shrunk by a margin."""
    raw = env.raw
    size = next((getattr(raw, a) for a in dir(raw) if a.endswith("table_full_size")), None)
    offset = next((getattr(raw, a) for a in dir(raw) if a.endswith("table_offset")), None)
    if size is None or offset is None:
        return None
    half = np.asarray(size[:2], dtype=np.float64) / 2.0 - margin_m
    center = np.asarray(offset[:2], dtype=np.float64)
    return center - half, center + half


def displace_objects(
    env: LiberoEnv,
    moves: dict[str, dict[str, Any]],
    *,
    min_delta_m: float,
    settle_steps: int = 20,
    max_drop_m: float = 0.05,
) -> dict[str, Any]:
    """moves: {object: {delta_xy: [dx, dy], yaw: rad}}. Raises Infeasible if an object fell or barely moved."""
    sim = env.sim
    before = {name: env.object_position(name).copy() for name in moves}
    bounds = table_bounds(env)
    for name, mv in moves.items():
        target = before[name][:2] + np.asarray(mv["delta_xy"], dtype=np.float64)
        if bounds is not None and (np.any(target < bounds[0]) or np.any(target > bounds[1])):
            raise Infeasible(f"{name} would leave the table at {target.round(3).tolist()}")
    for name, mv in moves.items():
        joint = env.object_joint(name)
        qpos = np.array(sim.data.get_joint_qpos(joint), dtype=np.float64).reshape(-1)
        if qpos.shape != (7,):
            raise Infeasible(f"{name} is not a free-joint object (qpos has {qpos.size} dofs)")
        qpos[0] += float(mv["delta_xy"][0])
        qpos[1] += float(mv["delta_xy"][1])
        qpos[2] += 0.005  # lift a hair so a rotated mesh does not start inside the table
        yaw = float(mv.get("yaw", 0.0))
        if yaw:
            qpos[3:7] = quat_multiply_wxyz(yaw_quaternion_wxyz(yaw), qpos[3:7])
        sim.data.set_joint_qpos(joint, qpos)
    env.forward()
    env.settle(settle_steps)
    realized: dict[str, Any] = {}
    for name in moves:
        after = env.object_position(name)
        delta = after[:2] - before[name][:2]
        dist = float(np.linalg.norm(delta))
        dropped = float(before[name][2] - after[2])
        realized[name] = {
            "delta_xy": [round(float(delta[0]), 4), round(float(delta[1]), 4)],
            "distance_m": round(dist, 4),
            "drop_m": round(dropped, 4),
        }
        if dropped > max_drop_m:
            raise Infeasible(f"{name} fell {dropped:.3f} m after displacement")
        if dist + 1e-6 < min_delta_m:
            raise Infeasible(f"{name} moved {dist:.3f} m, below the {min_delta_m:.3f} m minimum")
    return realized


def snapshot_lighting(env: LiberoEnv) -> dict[str, Any]:
    m = env.mj_model
    return {
        "active": m.light_active.copy(),
        "pos": m.light_pos.copy(),
        "dir": m.light_dir.copy(),
        "diffuse": m.light_diffuse.copy(),
        "ambient": m.light_ambient.copy(),
        "specular": m.light_specular.copy(),
        "headlight_diffuse": np.array(m.vis.headlight.diffuse, dtype=np.float64).copy(),
        "headlight_ambient": np.array(m.vis.headlight.ambient, dtype=np.float64).copy(),
    }


def restore_lighting(env: LiberoEnv, snap: dict[str, Any]) -> None:
    m = env.mj_model
    m.light_active[:] = snap["active"]
    m.light_pos[:] = snap["pos"]
    m.light_dir[:] = snap["dir"]
    m.light_diffuse[:] = snap["diffuse"]
    m.light_ambient[:] = snap["ambient"]
    m.light_specular[:] = snap["specular"]
    m.vis.headlight.diffuse[:] = snap["headlight_diffuse"]
    m.vis.headlight.ambient[:] = snap["headlight_ambient"]
    env.forward()  # light_xpos/xdir are recomputed by mj_forward; without it the renderer keeps the old lights


def _rotate_small(v: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate a direction by small angles about x, y, z (in that order)."""
    ax, ay, az = (float(a) for a in angles)
    cx, sx, cy, sy, cz, sz = (
        math.cos(ax),
        math.sin(ax),
        math.cos(ay),
        math.sin(ay),
        math.cos(az),
        math.sin(az),
    )
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    out = rz @ ry @ rx @ np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(out)
    return out / n if n > 1e-9 else out


def apply_lighting(env: LiberoEnv, lighting: dict[str, Any]) -> dict[str, Any]:
    """Apply a recorded draw. Returns what was applied (lights present in the model)."""
    m = env.mj_model
    n = int(m.nlight)
    applied = []
    for i, light in enumerate(lighting.get("lights", [])[:n]):
        m.light_active[i] = 1 if light.get("active", True) else 0
        m.light_diffuse[i] = np.clip(m.light_diffuse[i] * float(light["diffuse_scale"]), 0, 2)
        m.light_ambient[i] = np.clip(m.light_ambient[i] + float(light["ambient_add"]), 0, 1)
        m.light_specular[i] = np.clip(m.light_specular[i] * float(light["specular_scale"]), 0, 2)
        m.light_pos[i] = m.light_pos[i] + np.asarray(light["pos_jitter"], dtype=np.float64)
        m.light_dir[i] = _rotate_small(
            m.light_dir[i], np.asarray(light["dir_jitter"], dtype=np.float64)
        )
        applied.append(i)
    scale = float(lighting.get("headlight_scale", 1.0))
    m.vis.headlight.diffuse[:] = np.clip(np.asarray(m.vis.headlight.diffuse) * scale, 0, 2)
    env.forward()
    return {"lights_applied": applied, "model_lights": n}


def unit_moves(
    unit: dict[str, Any], env: LiberoEnv
) -> tuple[dict[str, dict[str, Any]], float] | None:
    """Translate a unit's perturbation record into object moves. None when nothing moves."""
    p = unit.get("perturbation", {})
    kind = p.get("kind")
    if kind == "pro_pose":
        return {p["target"]: {"delta_xy": p["delta_xy"], "yaw": p.get("yaw", 0.0)}}, float(
            p.get("min_delta_norm", 0.0)
        )
    if kind == "level":
        targets = env.movable_objects() if p.get("all_objects") else [p["target"]]
        return {t: {"delta_xy": p["delta_xy"], "yaw": p.get("yaw", 0.0)} for t in targets}, float(
            p.get("min_delta_m", 0.0)
        )
    return None
