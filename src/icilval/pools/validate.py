"""Simulator-side validation used while building a pool. Runs in the BPP environment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..sim import bddl as B
from ..sim.libero_env import LiberoEnv, load_init_states, save_init_states
from ..sim.perturb import Infeasible, displace_objects
from ..sim.perturb_math import direction_delta
from ..spec import Spec

log = logging.getLogger(__name__)


def valid_instances(env: LiberoEnv, states: np.ndarray, seed: int = 7) -> list[int]:
    """Initial states where the goal is not already satisfied."""
    out = []
    for i, s in enumerate(states):
        try:
            env.reset(seed, s)
        except Exception as exc:  # noqa: BLE001
            log.warning("instance %d: reset failed: %s", i, exc)
            continue
        if not env.success():
            out.append(i)
    return out


def displacement_slots(
    env: LiberoEnv,
    states: np.ndarray,
    moves_for: Any,
    min_delta_m: float,
    directions: int,
    seed: int = 7,
) -> list[int]:
    """Slots (instance * directions + k) where the displacement is feasible."""
    out = []
    failures: dict[str, int] = {}
    for i, s in enumerate(states):
        for k in range(directions):
            try:
                env.reset(seed, s)
                displace_objects(env, moves_for(k), min_delta_m=min_delta_m, settle_steps=10)
                if not env.success():
                    out.append(i * directions + k)
            except Infeasible as exc:
                key = str(exc).split(" at ")[0].split(" moved ")[0].split(" fell ")[0]
                failures[key] = failures.get(key, 0) + 1
            except Exception as exc:  # noqa: BLE001
                failures[f"{type(exc).__name__}: {exc}"] = (
                    failures.get(f"{type(exc).__name__}: {exc}", 0) + 1
                )
    if failures:
        log.info(
            "displacement rejections: %s", dict(sorted(failures.items(), key=lambda kv: -kv[1])[:4])
        )
    return out


def level_slots(
    env: LiberoEnv,
    states: np.ndarray,
    targets: list[str],
    radius_m: float,
    min_delta_m: float,
    directions: int,
) -> list[int]:
    def moves_for(k: int) -> dict[str, dict[str, Any]]:
        d = direction_delta(k, directions, radius_m)
        return {t: {"delta_xy": d, "yaw": 0.0} for t in targets}

    return displacement_slots(env, states, moves_for, min_delta_m, directions)


def pose_instances(
    env: LiberoEnv,
    states: np.ndarray,
    target: str,
    delta_xy: list[float],
    yaw: float,
    min_delta_m: float,
) -> list[int]:
    return displacement_slots(
        env, states, lambda k: {target: {"delta_xy": delta_xy, "yaw": yaw}}, min_delta_m, 1
    )


def regenerate_init_states(
    env: LiberoEnv, n: int, seed: int = 7, max_attempts: int | None = None
) -> np.ndarray:
    """Sample fresh initial states through LIBERO's own placement sampler; keep only stable ones."""
    states: list[np.ndarray] = []
    attempts = 0
    limit = max_attempts or n * 3
    while len(states) < n and attempts < limit:
        attempts += 1
        try:
            env.reset(seed + attempts, None)
            env.settle(10)
        except Exception as exc:  # noqa: BLE001
            log.debug("regenerate: reset failed: %s", exc)
            continue
        state = env.sim_state()
        if not np.all(np.isfinite(state)) or env.success():
            continue
        before = {o: env.object_position(o) for o in env.movable_objects()}
        env.settle(20)
        after = env.sim_state()
        if not np.all(np.isfinite(after)):
            continue
        if any(np.linalg.norm(env.object_position(o) - before[o]) > 0.02 for o in before):
            continue  # objects still falling or jittering: not a resting state
        states.append(env.sim_state())
    if len(states) < n:
        log.warning(
            "regenerate: only %d of %d stable initial states after %d attempts",
            len(states),
            n,
            attempts,
        )
    return np.stack(states) if states else np.zeros((0, 0))


def build_variant_env(
    bddl_path: Path, spec: Spec, scene_properties: dict[str, str] | None = None
) -> LiberoEnv | None:
    try:
        return LiberoEnv(bddl_path, spec, scene_properties=scene_properties)
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot build %s: %s", bddl_path.name, exc)
        return None


def render_differs(
    env_a: LiberoEnv, env_b: LiberoEnv, state: np.ndarray, threshold: float = 1.0
) -> bool:
    env_a.reset(7, state)
    env_b.reset(7, state)
    a, b = env_a.render().astype(int), env_b.render().astype(int)
    return float(np.abs(a - b).mean()) > threshold


def goal_from_bddl(path: Path) -> list[list[str]]:
    return B.goal_predicates(B.load(path))


__all__ = [
    "valid_instances",
    "level_slots",
    "pose_instances",
    "regenerate_init_states",
    "build_variant_env",
    "render_differs",
    "goal_from_bddl",
    "load_init_states",
    "save_init_states",
]
