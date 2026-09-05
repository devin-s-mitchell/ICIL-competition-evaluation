"""Deterministic unit lists.

A unit is one paired episode: axis, task (or variant), initial-state index,
prompt demonstration and seed. Both sides of a duel run the same list. The
list is a pure function of the pool and the duel id, so a third party holding
the pool can regenerate it from the published record.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..ids import unit_id, unit_seed
from ..rng import HashRng
from ..sim.lighting import sample_lighting
from ..spec import AXES, Spec
from .schema import Pool


@dataclass
class Unit:
    unit_id: str
    axis: str
    index: int
    task: str
    task_label: str
    variant: str | None
    instance: int
    demo: str
    seed: int
    perturbation: dict[str, Any]
    max_steps: int
    bddl: str
    init: str
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


def derive_units(pool: Pool, spec: Spec, duel: str, size: str | None = None) -> list[Unit]:
    per_axis = spec.units_per_axis(size)
    out: list[Unit] = []
    for axis in AXES:
        rng = HashRng(duel, axis)
        entries = pool.eligible(axis)
        if not entries:
            raise ValueError(f"pool has no eligible entries for axis {axis}")
        for index, entry in enumerate(_spread(per_axis, entries, rng)):
            task, variant = pool.resolve(axis, entry)
            n_init = variant.n_init if variant else task.n_init
            instance = rng.below(n_init)
            candidates = [d for d in task.demos if task.demo_init_index.get(d) != instance] or list(
                task.demos
            )
            if not candidates:
                raise ValueError(f"task {task.task_id} has no demonstrations")
            demo = candidates[rng.below(len(candidates))]
            seed = unit_seed(duel, axis, index)
            perturbation: dict[str, Any] = dict(task.perturbation)
            if variant:
                perturbation = {"kind": variant.kind, **variant.params}
            if axis == "environment":
                lighting = sample_lighting(
                    HashRng(duel, axis, index, "lighting"), spec.axis("environment")["lighting"]
                )
                perturbation = {
                    **perturbation,
                    "kind": perturbation.get("kind", "lighting"),
                    "lighting": lighting,
                }
            if "kind" not in perturbation:
                perturbation["kind"] = {"object": "object_swap", "composition": "chain"}.get(
                    axis, axis
                )
            out.append(
                Unit(
                    unit_id=unit_id(axis, index),
                    axis=axis,
                    index=index,
                    task=task.task_id,
                    task_label=task.label,
                    variant=variant.variant_id if variant else None,
                    instance=instance,
                    demo=demo,
                    seed=seed,
                    perturbation=perturbation,
                    max_steps=min(task.max_steps, spec.max_steps(axis))
                    if task.max_steps
                    else spec.max_steps(axis),
                    bddl=variant.bddl if variant else task.bddl,
                    init=variant.init if variant else task.init,
                    goal=task.goal,
                    steps=task.steps,
                )
            )
    return out
