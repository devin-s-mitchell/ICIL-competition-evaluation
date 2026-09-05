"""Run one side of a duel over a unit list. Resumable; runs where the simulator and torch live."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..canon import sha256_file
from ..model.bpp import BPPPolicy
from ..pools.demos import load_demo
from ..pools.schema import Pool
from ..sim.episode import run_episode
from ..sim.libero_env import LiberoEnv, load_init_states, scene_properties_of
from ..sim.video import VideoWriter
from ..spec import Spec

log = logging.getLogger(__name__)


def read_results(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        out[rec["unit_id"]] = rec
    return out


def run_side(
    *,
    side: str,
    model_dir: Path,
    arch_dir: Path,
    pool: Pool,
    units: list[dict[str, Any]],
    spec: Spec,
    out_dir: Path,
    device: str = "cuda",
    on_unit: Callable[[dict[str, Any]], None] | None = None,
    record_video: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    media_dir.mkdir(exist_ok=True)
    results_path = out_dir / "units.jsonl"
    done = read_results(results_path)
    t_start = time.monotonic()
    side_wall = float(spec.budgets["side_wall_seconds"])
    video_cfg = spec.media["video"]
    fps = int(video_cfg["fps"])

    policy = BPPPolicy(model_dir, arch_dir, spec, device=device)
    policy.load()
    summary = {
        "side": side,
        "units": len(units),
        "ran": 0,
        "resumed": len(done),
        "void": 0,
        "success": 0,
        "errors": 0,
        "load_seconds": policy.load_seconds,
    }
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    env: LiberoEnv | None = None
    env_key: tuple[str, str] | None = None
    init_cache: dict[str, np.ndarray] = {}
    try:
        # keep env switches rare: run units grouped by scene, in unit order within a group
        order = sorted(
            range(len(units)),
            key=lambda i: (
                units[i]["bddl"],
                json.dumps(scene_properties_of(units[i]), sort_keys=True),
                i,
            ),
        )
        for i in order:
            unit = units[i]
            if unit["unit_id"] in done:
                continue
            if time.monotonic() - t_start > side_wall:
                rec = {
                    "unit_id": unit["unit_id"],
                    "void": True,
                    "error": "side wall time exceeded",
                    "success": None,
                }
                _append(results_path, rec)
                summary["void"] += 1
                continue
            key = (unit["bddl"], json.dumps(scene_properties_of(unit), sort_keys=True))
            if env is None or env_key != key:
                if env is not None:
                    env.close()
                env = LiberoEnv(
                    pool.path(unit["bddl"]), spec, scene_properties=scene_properties_of(unit)
                )
                env_key = key
            if unit["init"] not in init_cache:
                init_cache[unit["init"]] = load_init_states(pool.path(unit["init"]))
            states = init_cache[unit["init"]]
            init_state = states[int(unit["instance"])]
            demo = load_demo(pool.path("demos") / f"{unit['demo']}.npz")
            clip = media_dir / f"{unit['unit_id']}.mp4"
            writer = VideoWriter(clip, fps, video_cfg) if record_video else None
            try:
                res = run_episode(
                    env, policy, unit, init_state, demo, spec, video=writer, executor=executor
                )
            finally:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception as exc:  # noqa: BLE001
                        log.warning("video encode failed for %s: %s", unit["unit_id"], exc)
            rec = {
                "unit_id": unit["unit_id"],
                "success": None if res.void else bool(res.success),
                "progress": res.progress,
                "first_step_done_at": res.first_step_done_at,
                "steps": res.steps,
                "model_errors": res.model_errors,
                "wall_s": round(res.wall_s, 2),
                "error": res.error,
                "void": res.void,
                "prompt_steps": res.prompt_steps,
                "prompt_chunks": res.prompt_chunks,
                "perturbation_applied": res.perturbation_applied,
                "video": None,
                "video_sha256": None,
            }
            if record_video and clip.exists() and clip.stat().st_size > 0 and not res.void:
                rec["video"] = str(clip.relative_to(out_dir))
                rec["video_sha256"] = sha256_file(clip)
            _append(results_path, rec)
            done[rec["unit_id"]] = rec
            summary["ran"] += 1
            summary["void"] += int(res.void)
            summary["success"] += int(bool(res.success))
            summary["errors"] += res.model_errors
            log.info(
                "%s %s: success=%s steps=%d wall=%.1fs%s",
                side,
                unit["unit_id"],
                res.success,
                res.steps,
                res.wall_s,
                f" error={res.error}" if res.error else "",
            )
            if on_unit is not None:
                on_unit(rec)
    finally:
        if env is not None:
            env.close()
        executor.shutdown(wait=False)
    summary["wall_seconds"] = round(time.monotonic() - t_start, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _append(path: Path, rec: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
