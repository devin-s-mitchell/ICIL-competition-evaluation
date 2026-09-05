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
    for i, s in enumerate(states):
        for k in range(directions):
            try:
                env.reset(seed, s)
                displace_objects(env, moves_for(k), min_delta_m=min_delta_m, settle_steps=10)
                if not env.success():
                    out.append(i * directions + k)
            except Infeasible:
                continue
            except Exception as exc:  # noqa: BLE001
                log.warning("instance %d dir %d: %s", i, k, exc)
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


def regenerate_init_states(env: LiberoEnv, n: int, seed: int = 7) -> np.ndarray:
    """Sample fresh initial states through LIBERO's own placement sampler."""
    states = []
    for i in range(n):
        env.reset(seed + i, None)
        env.settle(5)
        if env.success():
            continue
        states.append(env.sim_state())
    return np.stack(states) if states else np.zeros((0, 0))


def build_variant_env(bddl_path: Path, spec: Spec) -> LiberoEnv | None:
    try:
        return LiberoEnv(bddl_path, spec)
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
