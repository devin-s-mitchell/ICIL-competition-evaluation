"""The pool manifest: everything a duel draws from, content-addressed.

A pool is a directory holding `pool.json` plus `bddl/`, `init/` and `demos/`.
Tasks are the things a prompt demo exists for; variants are perturbed copies of
a task's scene (same goal, different BDDL + init states). The object and
composition axes draw tasks; spatial and environment draw variants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..canon import canonical_sha256

POOL_SCHEMA = 1


@dataclass
class PoolTask:
    task_id: str
    axis: str
    suite: str
    bddl: str
    language: str
    init: str
    n_init: int
    goal: list[list[str]]
    demos: list[str]
    max_steps: int
    steps: list[list[str]] = field(default_factory=list)
    demo_init_index: dict[str, int | None] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    perturbation: dict[str, Any] = field(default_factory=dict)
    instances: list[int] | None = None

    @property
    def valid_instances(self) -> list[int]:
        return list(self.instances) if self.instances is not None else list(range(self.n_init))

    @property
    def label(self) -> str:
        return self.language.strip().rstrip(".").capitalize() if self.language else self.task_id


@dataclass
class PoolVariant:
    variant_id: str
    axis: str
    base_task: str
    kind: str
    params: dict[str, Any]
    bddl: str
    init: str
    n_init: int
    validated: bool = True
    instances: list[int] | None = None

    @property
    def valid_instances(self) -> list[int]:
        return list(self.instances) if self.instances is not None else list(range(self.n_init))


@dataclass
class Pool:
    schema: int
    pool_version: str
    spec_version: int
    sources: dict[str, Any]
    tasks: dict[str, PoolTask]
    variants: dict[str, PoolVariant]
    axes: dict[str, dict[str, list[str]]]
    pool_id: str | None = None
    root: Path | None = None

    # -- (de)serialisation
    @classmethod
    def from_dict(cls, d: dict[str, Any], root: Path | None = None) -> Pool:
        tasks = {
            k: PoolTask(task_id=k, **{kk: vv for kk, vv in v.items() if kk != "task_id"})
            for k, v in d.get("tasks", {}).items()
        }
        variants = {
            k: PoolVariant(variant_id=k, **{kk: vv for kk, vv in v.items() if kk != "variant_id"})
            for k, v in d.get("variants", {}).items()
        }
        return cls(
            schema=int(d.get("schema", POOL_SCHEMA)),
            pool_version=str(d["pool_version"]),
            spec_version=int(d["spec_version"]),
            sources=dict(d.get("sources", {})),
            tasks=tasks,
            variants=variants,
            axes={
                a: {"eligible": list(v.get("eligible", []))} for a, v in d.get("axes", {}).items()
            },
            pool_id=d.get("pool_id"),
            root=root,
        )

    def to_dict(self, with_id: bool = True) -> dict[str, Any]:
        def task_dict(t: PoolTask) -> dict[str, Any]:
            return {
                "axis": t.axis,
                "suite": t.suite,
                "bddl": t.bddl,
                "language": t.language,
                "init": t.init,
                "n_init": t.n_init,
                "goal": t.goal,
                "steps": t.steps,
                "demos": t.demos,
                "demo_init_index": t.demo_init_index,
                "max_steps": t.max_steps,
                "provenance": t.provenance,
                "perturbation": t.perturbation,
                "instances": t.instances,
            }

        def variant_dict(v: PoolVariant) -> dict[str, Any]:
            return {
                "axis": v.axis,
                "base_task": v.base_task,
                "kind": v.kind,
                "params": v.params,
                "bddl": v.bddl,
                "init": v.init,
                "n_init": v.n_init,
                "validated": v.validated,
                "instances": v.instances,
            }

        d: dict[str, Any] = {
            "schema": self.schema,
            "pool_version": self.pool_version,
            "spec_version": self.spec_version,
            "sources": self.sources,
            "tasks": {k: task_dict(t) for k, t in sorted(self.tasks.items())},
            "variants": {k: variant_dict(v) for k, v in sorted(self.variants.items())},
            "axes": {a: {"eligible": sorted(v["eligible"])} for a, v in sorted(self.axes.items())},
        }
        if with_id:
            d["pool_id"] = self.pool_id
        return d

    def compute_id(self) -> str:
        return canonical_sha256(self.to_dict(with_id=False))

    def seal(self) -> str:
        self.pool_id = self.compute_id()
        return self.pool_id

    @classmethod
    def load(cls, root: str | Path) -> Pool:
        root = Path(root)
        doc = json.loads((root / "pool.json").read_text())
        pool = cls.from_dict(doc, root=root)
        if pool.pool_id and pool.pool_id != pool.compute_id():
            raise ValueError(f"{root}/pool.json: pool_id does not match its content")
        return pool

    def save(self, root: str | Path | None = None) -> Path:
        root = Path(root or self.root)
        root.mkdir(parents=True, exist_ok=True)
        if not self.pool_id:
            self.seal()
        path = root / "pool.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        self.root = root
        return path

    # -- lookups
    def eligible(self, axis: str) -> list[str]:
        return sorted(self.axes.get(axis, {}).get("eligible", []))

    def resolve(self, axis: str, entry: str) -> tuple[PoolTask, PoolVariant | None]:
        if entry in self.variants:
            v = self.variants[entry]
            return self.tasks[v.base_task], v
        return self.tasks[entry], None

    def path(self, rel: str) -> Path:
        if self.root is None:
            raise ValueError("pool has no root directory")
        return self.root / rel
