"""Deterministic unit lists.

A unit is one paired episode: skill, task (or variant), initial-state index,
prompt demonstration and seed, plus the perturbation it carries. Both sides
of a duel run the same list. The list is a pure function of the pool and the
duel id, so a third party holding the pool can regenerate it from the
published record.

A skill's units are spread evenly over its perturbation groups (in spec
order), and within a group over the eligible tasks or variants. The groups
are published on every unit and never scored on their own.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..ids import unit_id, unit_seed
from ..rng import HashRng
from ..sim.lighting import sample_lighting
from ..sim.perturb_math import direction_delta
from ..spec import Spec
from .schema import Pool, PoolTask


@dataclass
class Unit:
    unit_id: str
    skill: str
    kind: str
    index: int
    task: str
    task_label: str
    variant: str | None
    instance: int
    demo: str
    seed: int
    perturbation: dict[str, Any]
    max_steps: int
    bddl: str | None
    init: str | None
    goal: list[list[str]] = field(default_factory=list)
    steps: list[list[str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Unit:
        return cls(**d)


def _spread(n: int, entries: list[str], rng: HashRng) -> list[str]:
    """n picks over `entries`, no entry used more than ceil(n/len) times, order shuffled."""
    if not entries:
        return []
    order = rng.shuffled(entries)
    cap = math.ceil(n / len(order))
    picks: list[str] = []
    counts = dict.fromkeys(order, 0)
    i = 0
    while len(picks) < n:
        e = order[i % len(order)]
        if counts[e] < cap:
            picks.append(e)
            counts[e] += 1
        i += 1
    return picks


def group_sizes(n: int, groups: list[str]) -> dict[str, int]:
    """n units over the groups, as even as integers allow, the remainder to the first ones."""
    base, extra = divmod(n, len(groups))
    return {g: base + (1 if i < extra else 0) for i, g in enumerate(groups)}


# ---------------------------------------------------------------- drawing board instances
def draw_instance(task_id: str, instance: int, cfg: dict[str, Any], env: dict[str, Any]) -> dict:
    """The board angle and cursor start of one drawing instance: a pure function of the ids."""
    rng = HashRng("draw-instance", task_id, instance)
    lo, hi = (float(x) for x in cfg["angle_range_rad"])
    c_lo, c_hi = (int(x) for x in env["cursor_start_range_px"])
    return {
        "angle_rad": round(rng.uniform(lo, hi), 6),
        "cursor_px": [c_lo + rng.below(c_hi - c_lo + 1), c_lo + rng.below(c_hi - c_lo + 1)],
    }


def draw_demo_candidates(task: PoolTask, angle: float, min_delta: float) -> list[str]:
    """Demonstrations whose board angle differs from the instance's by at least min_delta."""
    angles = task.meta.get("demo_angles", {})
    return [d for d in task.demos if abs(float(angles.get(d, 0.0)) - angle) >= min_delta]


# ---------------------------------------------------------------- derivation
def derive_units(pool: Pool, spec: Spec, duel: str, size: str | None = None) -> list[Unit]:
    per_skill = spec.units_per_skill(size)
    out: list[Unit] = []
    for skill in spec.skills:
        groups = list(spec.perturbations(skill).keys())
        rng = HashRng(duel, skill)
        index = 0
        for group, n in group_sizes(per_skill, groups).items():
            entries = pool.eligible(skill, group)
            if n and not entries:
                raise ValueError(f"pool has no eligible entries for {skill}/{group}")
            for entry in _spread(n, entries, rng):
                out.append(_unit(pool, spec, duel, skill, group, index, entry, rng))
                index += 1
    return out


def _unit(
    pool: Pool, spec: Spec, duel: str, skill: str, group: str, index: int, entry: str, rng: HashRng
) -> Unit:
    task, variant = pool.resolve(entry)
    seed = unit_seed(duel, skill, index)
    cfg = spec.perturbation(skill, group)
    if spec.simulator(skill) == "draw":
        return _draw_unit(spec, skill, group, index, task, seed, cfg, rng)
    valid = (variant or task).valid_instances
    if not valid:
        raise ValueError(f"{entry} has no valid initial states")
    slot = valid[rng.below(len(valid))]
    directions = int(variant.params.get("directions", 0)) if variant else 0
    instance, direction = (slot // directions, slot % directions) if directions else (slot, None)
    candidates = [d for d in task.demos if task.demo_init_index.get(d) != instance] or list(
        task.demos
    )
    if not candidates:
        raise ValueError(f"task {task.task_id} has no demonstrations")
    demo = candidates[rng.below(len(candidates))]
    perturbation: dict[str, Any] = dict(task.perturbation)
    if variant:
        perturbation = {"kind": variant.kind, **variant.params}
        if direction is not None:
            perturbation["direction_index"] = direction
            perturbation["delta_xy"] = direction_delta(
                direction, directions, float(variant.params["radius_m"])
            )
            perturbation["yaw"] = round(
                rng.uniform(-1, 1) * float(variant.params.get("yaw_max_rad", 0.0)), 4
            )
            perturbation.pop("directions", None)
    if group == "environment":
        lighting = sample_lighting(HashRng(duel, skill, index, "lighting"), cfg["lighting"])
        perturbation = {
            **perturbation,
            "kind": perturbation.get("kind", "lighting"),
            "lighting": lighting,
        }
    if "kind" not in perturbation:
        perturbation["kind"] = task.kind if task.kind != "base" else group
    return Unit(
        unit_id=unit_id(spec.skill_code(skill), index),
        skill=skill,
        kind=group,
        index=index,
        task=task.task_id,
        task_label=task.label,
        variant=variant.variant_id if variant else None,
        instance=instance,
        demo=demo,
        seed=seed,
        perturbation=perturbation,
        max_steps=min(task.max_steps, spec.max_steps(skill))
        if task.max_steps
        else spec.max_steps(skill),
        bddl=variant.bddl if variant else task.bddl,
        init=variant.init if variant else task.init,
        goal=task.goal,
        steps=task.steps,
    )


def _draw_unit(
    spec: Spec,
    skill: str,
    group: str,
    index: int,
    task: PoolTask,
    seed: int,
    cfg: dict[str, Any],
    rng: HashRng,
) -> Unit:
    env = spec.env(skill)
    min_delta = float(cfg["min_delta_rad"])
    valid = task.valid_instances
    if not valid:
        raise ValueError(f"{task.task_id} has no instances")
    # an instance is usable only if some demonstration was drawn at a different enough angle
    usable = [
        i
        for i in valid
        if draw_demo_candidates(
            task, draw_instance(task.task_id, i, cfg, env)["angle_rad"], min_delta
        )
    ]
    if not usable:
        raise ValueError(f"{task.task_id}: no instance has a demonstration far enough in angle")
    instance = usable[rng.below(len(usable))]
    state = draw_instance(task.task_id, instance, cfg, env)
    candidates = draw_demo_candidates(task, state["angle_rad"], min_delta)
    demo = candidates[rng.below(len(candidates))]
    demo_angle = float(task.meta.get("demo_angles", {}).get(demo, 0.0))
    perturbation = {
        "kind": "rotation",
        "angle_rad": state["angle_rad"],
        "demo_angle_rad": round(demo_angle, 6),
        "delta_rad": round(abs(state["angle_rad"] - demo_angle), 6),
        "cursor_px": state["cursor_px"],
    }
    return Unit(
        unit_id=unit_id(spec.skill_code(skill), index),
        skill=skill,
        kind=group,
        index=index,
        task=task.task_id,
        task_label=task.label,
        variant=None,
        instance=instance,
        demo=demo,
        seed=seed,
        perturbation=perturbation,
        max_steps=min(task.max_steps, spec.max_steps(skill))
        if task.max_steps
        else spec.max_steps(skill),
        bddl=None,
        init=None,
        goal=[],
        steps=[],
    )
