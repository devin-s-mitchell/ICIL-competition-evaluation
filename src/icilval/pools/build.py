"""`icilval pools build`: turn raw benchmark assets into a content-addressed pool.

Stages (each idempotent, each appends to pool.json):
  base         LIBERO spatial/goal/object/10 tasks that are pick-and-place: bddl, init npz, k demos
  spatial      LIBERO-PRO swap + pose variants, our level ladder; validated per instance/slot
  environment  table swaps (init states regenerated) and pure lighting
  object       LIBERO-Gen novel pairings (spatial combinations + first-step novelties)
  draw         DrawAnything-Sim: the human-drawn evaluation set (and generated tasks, see build_draw)
  finalize     eligibility per skill and perturbation group, pool_id

Which tasks count as pick-and-place is `spec.json`'s `skills.pick_and_place.task_filter`:
BPP's one Grasp stage then one Place stage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..sim import bddl as B
from ..spec import Spec
from . import validate as V
from .demos import import_hdf5
from .schema import POOL_SCHEMA, Pool, PoolTask, PoolVariant
from .sources import (
    LIBERO_GOAL_ORIGINALS,
    LIBERO_SUITES,
    SUITE_MAX_STEPS,
    Sources,
    convert_init,
    copy_bddl,
)

log = logging.getLogger(__name__)

TABLE_INIT_STATES = 20
LIBERO_SKILL = "pick_and_place"
DRAW_SKILL = "draw_anything"


def open_pool(out: Path, spec: Spec, version: str) -> Pool:
    if (out / "pool.json").exists():
        pool = Pool.load(out)
        pool.pool_id = None
        return pool
    return Pool(
        schema=POOL_SCHEMA,
        pool_version=version,
        spec_version=spec.version,
        sources={},
        tasks={},
        variants={},
        skills={s: {g: {"eligible": []} for g in spec.perturbations(s)} for s in spec.skills},
        root=out,
    )


# ---------------------------------------------------------------- the pick-and-place filter
def is_pick_and_place(goal: list[list[str]], language: str, filt: dict[str, Any]) -> bool:
    """BPP's definition: one Grasp then one Place - a single On/In goal, no other stage."""
    if not goal or len(goal) > int(filt["max_goal_predicates"]):
        return False
    allowed = {p.lower() for p in filt["goal_predicates"]}
    if not {g[0].lower() for g in goal} <= allowed:
        return False
    lang = " ".join(language.lower().split())
    return not any(verb in lang for verb in filt["exclude_verbs"])


def _task_from_bddl(
    task_id: str,
    skill: str,
    kind: str,
    suite: str,
    bddl_rel: str,
    init_rel: str,
    n_init: int,
    demos: list[str],
    max_steps: int,
    pool: Pool,
    **extra: Any,
) -> PoolTask:
    tree = B.load(pool.path(bddl_rel))
    goal = B.goal_predicates(tree)
    task = PoolTask(
        task_id=task_id,
        skill=skill,
        kind=kind,
        suite=suite,
        bddl=bddl_rel,
        language=B.language(tree),
        init=init_rel,
        n_init=n_init,
        goal=goal,
        demos=demos,
        max_steps=max_steps,
        steps=goal,
        **extra,
    )
    pool.tasks[task_id] = task
    return task


def _import_demos(h5: Path, pool: Pool, task_id: str, k: int) -> list[str]:
    out_dir = pool.path("demos") / task_id
    if out_dir.exists() and len(list(out_dir.glob("demo_*.npz"))) >= min(k, 1):
        return sorted(f"{task_id}/{p.stem}" for p in out_dir.glob("demo_*.npz"))
    metas = import_hdf5(h5, out_dir, task_id, k)
    return [m.demo_id for m in metas]


# ---------------------------------------------------------------- base
def stage_base(
    pool: Pool,
    spec: Spec,
    src: Sources,
    *,
    skill: str = LIBERO_SKILL,
    suites: tuple[str, ...] = LIBERO_SUITES,
    limit: int | None = None,
) -> None:
    k = int(spec.pools["demos_per_task"])
    filt = spec.skill(skill)["task_filter"]
    for suite in suites:
        bddls = sorted((src.libero_root / "bddl_files" / suite).glob("*.bddl"))[:limit]
        for bddl in bddls:
            name = bddl.stem
            task_id = f"{suite}/{name}"
            if task_id in pool.tasks:
                continue
            tree = B.load(bddl)
            if not is_pick_and_place(B.goal_predicates(tree), B.language(tree), filt):
                log.info("skip %s: not pick-and-place", task_id)
                continue
            h5 = src.libero_datasets / suite / f"{name}_demo.hdf5"
            init_src = src.libero_root / "init_files" / suite / f"{name}.pruned_init"
            if not h5.exists() or not init_src.exists():
                log.warning("skip %s: missing demos or init", task_id)
                continue
            bddl_rel, init_rel = f"bddl/{suite}/{name}.bddl", f"init/{suite}/{name}.npz"
            copy_bddl(bddl, pool.path(bddl_rel))
            n_init = convert_init(init_src, pool.path(init_rel))
            demos = _import_demos(h5, pool, task_id, k)
            _task_from_bddl(
                task_id,
                skill,
                "base",
                suite,
                bddl_rel,
                init_rel,
                n_init,
                demos,
                SUITE_MAX_STEPS[suite],
                pool,
                provenance={"source": "LIBERO", "suite": suite, "demos": h5.name},
            )
            log.info("base %s: %d inits, %d demos", task_id, n_init, len(demos))
    pool.sources["libero"] = {"root": str(src.libero_root), "datasets": str(src.libero_datasets)}
    pool.save()


# ---------------------------------------------------------------- spatial
def stage_spatial(
    pool: Pool,
    spec: Spec,
    src: Sources,
    *,
    skill: str = LIBERO_SKILL,
    suites: tuple[str, ...] = LIBERO_SUITES,
    limit: int | None = None,
    validate: bool = True,
) -> None:
    cfg = spec.perturbation(skill, "spatial")
    levels: dict[str, dict[str, Any]] = cfg["levels"]
    directions = 8
    for suite in suites:
        tasks = [
            t
            for t in sorted(pool.tasks)
            if t.startswith(suite + "/") and pool.tasks[t].skill == skill
        ][:limit]
        for task_id in tasks:
            task = pool.tasks[task_id]
            name = task_id.split("/", 1)[1]
            base_states = V.load_init_states(pool.path(task.init))
            env = V.build_variant_env(pool.path(task.bddl), spec, skill) if validate else None
            movable = env.movable_objects() if env else []
            tree = B.load(pool.path(task.bddl))
            interest = [o for o in B.obj_of_interest(tree) if o in B.objects(tree)]
            target = next(
                (o for o in interest if not movable or o in movable),
                interest[0] if interest else None,
            )

            # LIBERO-PRO swap: shipped bddl + init
            vid = f"{task_id}#pro_swap"
            swap_bddl = src.libero_pro / "bddl_files" / f"{suite}_swap" / f"{name}.bddl"
            swap_init = src.libero_pro / "init_files" / f"{suite}_swap" / f"{name}.pruned_init"
            if vid not in pool.variants and swap_bddl.exists() and swap_init.exists():
                bddl_rel, init_rel = (
                    f"bddl/{suite}_swap/{name}.bddl",
                    f"init/{suite}_swap/{name}.npz",
                )
                copy_bddl(swap_bddl, pool.path(bddl_rel))
                n = convert_init(swap_init, pool.path(init_rel))
                inst = None
                if validate:
                    venv = V.build_variant_env(pool.path(bddl_rel), spec, skill)
                    if venv is None:
                        n = 0
                    else:
                        inst = V.valid_instances(venv, V.load_init_states(pool.path(init_rel)))
                        venv.close()
                if n:
                    pool.variants[vid] = PoolVariant(
                        variant_id=vid,
                        skill=skill,
                        base_task=task_id,
                        kind="pro_swap",
                        params={"source": "LIBERO-PRO position (swap)"},
                        bddl=bddl_rel,
                        init=init_rel,
                        n_init=n,
                        validated=validate,
                        instances=inst,
                    )
                    log.info("spatial %s: %s valid", vid, len(inst) if inst is not None else n)

            # LIBERO-PRO initial pose: delta on the target object, applied at reset
            vid = f"{task_id}#pro_pose"
            pose_bddl = (
                src.libero_pro
                / "bddl_files"
                / "07_initial_pose_position_angle"
                / "bddl"
                / suite
                / f"{name}.bddl"
            )
            if vid not in pool.variants and pose_bddl.exists():
                pose = B.libero_pro_initial_pose(B.load(pose_bddl))
                if pose:
                    inst = None
                    if env is not None:
                        inst = V.pose_instances(
                            env,
                            base_states,
                            pose["target"],
                            pose["delta_xy"],
                            pose["yaw"],
                            pose["min_delta_norm"],
                        )
                    pool.variants[vid] = PoolVariant(
                        variant_id=vid,
                        skill=skill,
                        base_task=task_id,
                        kind="pro_pose",
                        params={**pose, "source": "LIBERO-PRO 07_initial_pose_position_angle"},
                        bddl=task.bddl,
                        init=task.init,
                        n_init=task.n_init,
                        validated=validate,
                        instances=inst,
                    )
                    log.info(
                        "spatial %s: %s valid", vid, len(inst) if inst is not None else task.n_init
                    )

            # level ladder
            if target:
                for level, lv in levels.items():
                    vid = f"{task_id}#{level}"
                    if vid in pool.variants:
                        continue
                    targets = movable if lv.get("all_objects") and movable else [target]
                    slots = None
                    if env is not None:
                        slots = V.level_slots(
                            env,
                            base_states,
                            targets,
                            float(lv["radius_m"]),
                            float(lv["min_delta_m"]),
                            directions,
                        )
                    params = {
                        "level": level,
                        "radius_m": lv["radius_m"],
                        "min_delta_m": lv["min_delta_m"],
                        "all_objects": bool(lv.get("all_objects")),
                        "target": target,
                        "yaw_max_rad": cfg["yaw_max_rad"],
                        "directions": directions,
                    }
                    pool.variants[vid] = PoolVariant(
                        variant_id=vid,
                        skill=skill,
                        base_task=task_id,
                        kind="level",
                        params=params,
                        bddl=task.bddl,
                        init=task.init,
                        n_init=task.n_init * directions,
                        validated=validate,
                        instances=slots,
                    )
                    log.info(
                        "spatial %s: %s valid slots",
                        vid,
                        len(slots) if slots is not None else task.n_init * directions,
                    )
            if env is not None:
                env.close()
            pool.save()


# ---------------------------------------------------------------- environment
def stage_environment(
    pool: Pool,
    spec: Spec,
    src: Sources,
    *,
    skill: str = LIBERO_SKILL,
    suites: tuple[str, ...] = LIBERO_SUITES,
    limit: int | None = None,
    validate: bool = True,
) -> None:
    cfg = spec.perturbation(skill, "environment")
    for suite in suites:
        tasks = [
            t
            for t in sorted(pool.tasks)
            if t.startswith(suite + "/") and pool.tasks[t].skill == skill
        ][:limit]
        for task_id in tasks:
            task = pool.tasks[task_id]
            name = task_id.split("/", 1)[1]
            base_tree = B.load(pool.path(task.bddl))
            base_states = V.load_init_states(pool.path(task.init))
            base_env = V.build_variant_env(pool.path(task.bddl), spec, skill) if validate else None

            # table swaps: scene moves onto another LIBERO table; init states regenerated
            for table in cfg["tables"]:
                vid = f"{task_id}#table:{table}"
                if vid in pool.variants:
                    continue
                tree = json.loads(json.dumps(base_tree))
                try:
                    old = B.swap_table(tree, table)
                except (KeyError, ValueError) as exc:
                    log.warning("%s: %s", vid, exc)
                    continue
                if old == table:
                    continue
                bddl_rel, init_rel = (
                    f"bddl/{suite}_table_{table}/{name}.bddl",
                    f"init/{suite}_table_{table}/{name}.npz",
                )
                pool.path(bddl_rel).parent.mkdir(parents=True, exist_ok=True)
                B.dump(tree, pool.path(bddl_rel))
                inst = None
                n = 0
                if validate:
                    venv = V.build_variant_env(pool.path(bddl_rel), spec, skill)
                    if venv is None:
                        continue
                    states = V.regenerate_init_states(venv, TABLE_INIT_STATES)
                    venv.close()
                    if states.size == 0:
                        log.warning("%s: no initial states could be sampled", vid)
                        continue
                    V.save_init_states(pool.path(init_rel), states)
                    n = int(states.shape[0])
                    inst = list(range(n))
                pool.variants[vid] = PoolVariant(
                    variant_id=vid,
                    skill=skill,
                    base_task=task_id,
                    kind="table",
                    params={"table": table, "from": old},
                    bddl=bddl_rel,
                    init=init_rel,
                    n_init=n,
                    validated=validate,
                    instances=inst,
                )
                log.info("environment %s: %d inits", vid, n)

            # lighting only
            vid = f"{task_id}#lighting"
            if vid not in pool.variants:
                inst = V.valid_instances(base_env, base_states) if base_env is not None else None
                pool.variants[vid] = PoolVariant(
                    variant_id=vid,
                    skill=skill,
                    base_task=task_id,
                    kind="lighting",
                    params={},
                    bddl=task.bddl,
                    init=task.init,
                    n_init=task.n_init,
                    validated=validate,
                    instances=inst,
                )
            if base_env is not None:
                base_env.close()
            pool.save()


# ---------------------------------------------------------------- object (LIBERO-Gen)
def _gen_task(
    pool: Pool,
    spec: Spec,
    split_root: Path,
    split: str,
    name: str,
    task_id: str,
    skill: str,
    kind: str,
    max_steps: int,
    perturbation: dict[str, Any],
    validate: bool,
) -> PoolTask | None:
    bddl = split_root / "bddl_files" / split / f"{name}.bddl"
    init = split_root / "init_files" / split / f"{name}.pruned_init"
    h5 = split_root / "demonstration_data" / split / f"{name}_demo.hdf5"
    if not (bddl.exists() and init.exists() and h5.exists()):
        log.warning("skip %s: missing bddl/init/demos", task_id)
        return None
    tree = B.load(bddl)
    if not is_pick_and_place(
        B.goal_predicates(tree), B.language(tree), spec.skill(skill)["task_filter"]
    ):
        log.info("skip %s: not pick-and-place", task_id)
        return None
    group = task_id.split("/", 1)[0]
    bddl_rel, init_rel = f"bddl/{group}/{name}.bddl", f"init/{group}/{name}.npz"
    copy_bddl(bddl, pool.path(bddl_rel))
    n_init = convert_init(init, pool.path(init_rel))
    demos = _import_demos(h5, pool, task_id, int(spec.pools["demos_per_task"]))
    task = _task_from_bddl(
        task_id,
        skill,
        kind,
        split,
        bddl_rel,
        init_rel,
        n_init,
        demos,
        max_steps,
        pool,
        provenance={"source": "LIBERO-Gen (public)", "split": split, "demos": h5.name},
        perturbation=perturbation,
    )
    if validate:
        env = V.build_variant_env(pool.path(bddl_rel), spec, skill)
        if env is None:
            del pool.tasks[task_id]
            return None
        task.instances = V.valid_instances(env, V.load_init_states(pool.path(init_rel)))
        env.close()
    return task


def _swap_from_goal(goal: list[list[str]]) -> dict[str, Any]:
    for p in goal:
        if len(p) == 3 and p[0].lower() in ("on", "in"):
            return {
                "kind": "object_swap",
                "operator": "place_in" if p[0].lower() == "in" else "place_on",
                "object": p[1],
                "target": p[2],
            }
    return {"kind": "object_swap"}


def stage_object(
    pool: Pool,
    spec: Spec,
    src: Sources,
    *,
    skill: str = LIBERO_SKILL,
    limit: int | None = None,
    validate: bool = True,
) -> None:
    combo = "libero_spatial_selected_combinations_view"
    names = sorted(
        p.stem for p in (src.gen_spatial_combination / "bddl_files" / combo).glob("*.bddl")
    )[:limit]
    for name in names:
        task_id = f"libero_gen_spatial_combination/{name}"
        if task_id in pool.tasks:
            continue
        goal = V.goal_from_bddl(src.gen_spatial_combination / "bddl_files" / combo / f"{name}.bddl")
        t = _gen_task(
            pool,
            spec,
            src.gen_spatial_combination,
            combo,
            name,
            task_id,
            skill,
            "object_swap",
            spec.max_steps(skill),
            {**_swap_from_goal(goal), "source_split": combo},
            validate,
        )
        if t:
            log.info("object %s: %d demos", task_id, len(t.demos))
    first = "libero_goal_chain_firststep_view"
    names = sorted(p.stem for p in (src.gen_goal_chain / "bddl_files" / first).glob("*.bddl"))
    names = [n for n in names if n not in LIBERO_GOAL_ORIGINALS][:limit]
    for name in names:
        task_id = f"libero_gen_goal_firststep/{name}"
        if task_id in pool.tasks:
            continue
        goal = V.goal_from_bddl(src.gen_goal_chain / "bddl_files" / first / f"{name}.bddl")
        t = _gen_task(
            pool,
            spec,
            src.gen_goal_chain,
            first,
            name,
            task_id,
            skill,
            "object_swap",
            SUITE_MAX_STEPS["libero_goal"],
            {**_swap_from_goal(goal), "source_split": first},
            validate,
        )
        if t:
            log.info("object %s: %d demos", task_id, len(t.demos))
    pool.save()


# ---------------------------------------------------------------- finalize
def _draw_usable(task: PoolTask, spec: Spec) -> bool:
    from .units import draw_demo_candidates, draw_instance

    cfg = spec.perturbation(task.skill, "rotation")
    env = spec.env(task.skill)
    return any(
        draw_demo_candidates(
            task, draw_instance(task.task_id, i, cfg, env)["angle_rad"], float(cfg["min_delta_rad"])
        )
        for i in task.valid_instances
    )


def finalize(pool: Pool, spec: Spec) -> dict[str, dict[str, int]]:
    skills: dict[str, dict[str, dict[str, list[str]]]] = {}
    for skill in spec.skills:
        skills[skill] = {}
        for group, cfg in spec.perturbations(skill).items():
            eligible: list[str] = []
            for kind in cfg.get("variant_kinds", []) or []:
                eligible += [
                    vid
                    for vid, v in pool.variants.items()
                    if v.skill == skill and v.kind == kind and v.valid_instances
                ]
            for kind in cfg.get("task_kinds", []) or []:
                for tid, t in pool.tasks.items():
                    if t.skill != skill or t.kind != kind or not t.valid_instances or not t.demos:
                        continue
                    if spec.simulator(skill) == "draw" and not _draw_usable(t, spec):
                        log.warning("%s: no instance far enough from every demonstration", tid)
                        continue
                    eligible.append(tid)
            skills[skill][group] = {"eligible": sorted(set(eligible))}
    pool.skills = skills
    pool.spec_version = spec.version
    pool.seal()
    pool.save()
    return {s: {g: len(e["eligible"]) for g, e in groups.items()} for s, groups in skills.items()}


def verify_pool(root: Path, spec: Spec | None = None) -> list[str]:
    errors: list[str] = []
    pool = Pool.load(root)
    for tid, t in pool.tasks.items():
        for rel in (t.bddl, t.init):
            if rel and not pool.path(rel).exists():
                errors.append(f"{tid}: missing {rel}")
        for d in t.demos:
            if not (pool.path("demos") / f"{d}.npz").exists():
                errors.append(f"{tid}: missing demo {d}")
        if t.bddl:
            try:
                B.load(pool.path(t.bddl))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{tid}: bddl unparsable: {exc}")
        if spec is not None and t.skill not in spec.skills:
            errors.append(f"{tid}: unknown skill {t.skill}")
    for vid, v in pool.variants.items():
        if v.base_task not in pool.tasks:
            errors.append(f"{vid}: base task missing")
        for rel in (v.bddl, v.init):
            if not pool.path(rel).exists():
                errors.append(f"{vid}: missing {rel}")
    groups = (
        {s: list(spec.perturbations(s)) for s in spec.skills}
        if spec is not None
        else {s: list(g) for s, g in pool.skills.items()}
    )
    for skill, names in groups.items():
        for group in names:
            if not pool.eligible(skill, group):
                errors.append(f"{skill}/{group}: nothing eligible")
            for e in pool.eligible(skill, group):
                if e not in pool.variants and e not in pool.tasks:
                    errors.append(f"{skill}/{group}: unknown entry {e}")
    return errors


def summary(pool: Pool) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    for v in pool.variants.values():
        by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
    return {
        "pool_id": pool.pool_id,
        "tasks": {s: len(pool.tasks_of(s)) for s in pool.skills} | {"total": len(pool.tasks)},
        "variants": by_kind,
        "eligible": {
            s: {g: len(e["eligible"]) for g, e in groups.items()}
            for s, groups in pool.skills.items()
        },
        "demos": sum(len(t.demos) for t in pool.tasks.values()),
    }


def init_counts(pool: Pool) -> dict[str, int]:
    return {vid: len(v.valid_instances) for vid, v in pool.variants.items()}


__all__ = [
    "open_pool",
    "is_pick_and_place",
    "stage_base",
    "stage_spatial",
    "stage_environment",
    "stage_object",
    "finalize",
    "verify_pool",
    "summary",
    "np",
]
