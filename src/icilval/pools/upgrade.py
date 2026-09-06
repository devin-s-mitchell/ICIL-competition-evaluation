"""`icilval pools upgrade`: a schema-1 pool (four axes) -> a schema-2 pool (skills).

The LIBERO artifacts a pick-and-place task needs - its BDDL, initial states, demonstrations
and every validated variant - are copied unchanged, so nothing is re-simulated; tasks that
are not pick-and-place under `spec.json`'s filter, and every composition task, are dropped.
The drawing skill is then added with `pools build --stage draw finalize` on the same directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from ..spec import Spec
from .build import LIBERO_SKILL, is_pick_and_place
from .schema import POOL_SCHEMA, Pool, PoolTask, PoolVariant

log = logging.getLogger(__name__)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def upgrade_pool(old_root: Path, out: Path, spec: Spec, *, skill: str = LIBERO_SKILL) -> Pool:
    doc: dict[str, Any] = json.loads((old_root / "pool.json").read_text())
    if int(doc.get("schema", 1)) != 1:
        raise ValueError(f"{old_root}: expected a schema-1 pool, got {doc.get('schema')}")
    filt = spec.skill(skill)["task_filter"]
    pool = Pool(
        schema=POOL_SCHEMA,
        pool_version=str(spec.pools["version"]),
        spec_version=spec.version,
        sources={
            **doc.get("sources", {}),
            "upgraded_from": {
                "pool_id": doc.get("pool_id"),
                "pool_version": doc.get("pool_version"),
            },
        },
        tasks={},
        variants={},
        skills={s: {g: {"eligible": []} for g in spec.perturbations(s)} for s in spec.skills},
        root=out,
    )
    out.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    for tid, t in sorted(doc["tasks"].items()):
        axis = t.get("axis")
        if axis == "composition" or t.get("perturbation", {}).get("kind") == "chain":
            continue
        if not is_pick_and_place(t["goal"], t["language"], filt):
            log.info("drop %s: not pick-and-place", tid)
            continue
        pool.tasks[tid] = PoolTask(
            task_id=tid,
            skill=skill,
            kind="base" if axis == "base" else "object_swap",
            suite=t["suite"],
            bddl=t["bddl"],
            language=t["language"],
            init=t["init"],
            n_init=int(t["n_init"]),
            goal=t["goal"],
            steps=t.get("steps", t["goal"]),
            demos=list(t["demos"]),
            max_steps=min(int(t["max_steps"]), spec.max_steps(skill)),
            demo_init_index=dict(t.get("demo_init_index", {})),
            provenance=dict(t.get("provenance", {})),
            perturbation=dict(t.get("perturbation", {})),
            instances=t.get("instances"),
        )
        kept.append(tid)
        _copy(old_root / t["bddl"], out / t["bddl"])
        _copy(old_root / t["init"], out / t["init"])
        for d in t["demos"]:
            _copy(old_root / "demos" / f"{d}.npz", out / "demos" / f"{d}.npz")
    for vid, v in sorted(doc.get("variants", {}).items()):
        if v["base_task"] not in pool.tasks:
            continue
        pool.variants[vid] = PoolVariant(
            variant_id=vid,
            skill=skill,
            base_task=v["base_task"],
            kind=v["kind"],
            params=dict(v.get("params", {})),
            bddl=v["bddl"],
            init=v["init"],
            n_init=int(v["n_init"]),
            validated=bool(v.get("validated", True)),
            instances=v.get("instances"),
        )
        _copy(old_root / v["bddl"], out / v["bddl"])
        _copy(old_root / v["init"], out / v["init"])
    log.info(
        "kept %d of %d tasks and %d variants", len(kept), len(doc["tasks"]), len(pool.variants)
    )
    pool.pool_id = None
    pool.save()
    return pool
