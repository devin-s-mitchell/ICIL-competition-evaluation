"""The contract: `spec.json`, read once, validated, never duplicated as literals."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .canon import canonical_sha256

AXES: tuple[str, ...] = ("spatial", "environment", "object", "composition")
SPEC_ENV = "ICILVAL_SPEC"
SCHEMA_ENV = "ICILVAL_STORE_SCHEMA"


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "spec.json").exists():
            return parent
    return None


def _resolve(name: str, env: str) -> Path:
    explicit = os.environ.get(env)
    if explicit:
        return Path(explicit)
    packaged = Path(__file__).resolve().parent / "data" / name
    if packaged.exists():
        return packaged
    root = _repo_root()
    if root and (root / name).exists():
        return root / name
    raise FileNotFoundError(f"{name} not found; set {env}")


def spec_path() -> Path:
    return _resolve("spec.json", SPEC_ENV)


def schema_path() -> Path:
    return _resolve("store-schema.json", SCHEMA_ENV)


def validate_spec(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def need(path: str, cond: bool) -> None:
        if not cond:
            errors.append(path)

    need("spec_version:int", isinstance(doc.get("spec_version"), int))
    track = doc.get("track") or {}
    need("track.id", isinstance(track.get("id"), str) and bool(track.get("id")))
    need("track.k_demos==1", track.get("k_demos") == 1)
    need("track.language==none", track.get("language") == "none")
    axes = doc.get("axes") or {}
    need("axes keys", tuple(axes.keys()) == AXES)
    for axis, a in axes.items():
        need(f"axes.{axis}.max_steps", isinstance(a.get("max_steps"), int) and a["max_steps"] > 0)
    duel = doc.get("duel") or {}
    sizes = duel.get("sizes") or {}
    need("duel.default_size in sizes", duel.get("default_size") in sizes)
    for name, s in sizes.items():
        n = s.get("units_per_axis")
        need(f"duel.sizes.{name}.units_per_axis>=1", isinstance(n, int) and n >= 1)
    margin = duel.get("score_margin")
    need("duel.score_margin in [0,100]", isinstance(margin, (int, float)) and 0 <= margin <= 100)
    void = duel.get("max_void_fraction")
    need("duel.max_void_fraction in [0,1]", isinstance(void, (int, float)) and 0 <= void <= 1)
    store = doc.get("store") or {}
    need(
        "store.index_lines_per_part>=1",
        isinstance(store.get("index_lines_per_part"), int) and store["index_lines_per_part"] >= 1,
    )
    need("store.media_bucket_hex in 1..4", store.get("media_bucket_hex") in (1, 2, 3, 4))
    model = doc.get("model") or {}
    need(
        "model.required_files",
        isinstance(model.get("required_files"), list) and model["required_files"],
    )
    need("model.max_repo_bytes", isinstance(model.get("max_repo_bytes"), int))
    return errors


@dataclass(frozen=True)
class Spec:
    raw: dict[str, Any]
    path: Path
    fingerprint: str

    # -- track / axes
    @property
    def version(self) -> int:
        return int(self.raw["spec_version"])

    @property
    def track_id(self) -> str:
        return str(self.raw["track"]["id"])

    @property
    def track(self) -> dict[str, Any]:
        return self.raw["track"]

    @property
    def axes(self) -> tuple[str, ...]:
        return AXES

    def axis(self, name: str) -> dict[str, Any]:
        return self.raw["axes"][name]

    def max_steps(self, axis: str) -> int:
        return int(self.raw["axes"][axis]["max_steps"])

    # -- duel
    @property
    def duel(self) -> dict[str, Any]:
        return self.raw["duel"]

    @property
    def default_size(self) -> str:
        return str(self.raw["duel"]["default_size"])

    @property
    def sizes(self) -> tuple[str, ...]:
        return tuple(self.raw["duel"]["sizes"].keys())

    def size_of(self, size: str | None) -> str:
        return size if size in self.raw["duel"]["sizes"] else self.default_size

    def units_per_axis(self, size: str | None = None) -> int:
        return int(self.raw["duel"]["sizes"][self.size_of(size)]["units_per_axis"])

    def units_per_duel(self, size: str | None = None) -> int:
        return self.units_per_axis(size) * len(AXES)

    @property
    def score_margin(self) -> float:
        return float(self.raw["duel"]["score_margin"])

    @property
    def max_void_fraction(self) -> float:
        return float(self.raw["duel"]["max_void_fraction"])

    # -- the rest
    @property
    def budgets(self) -> dict[str, Any]:
        return self.raw["budgets"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def media(self) -> dict[str, Any]:
        return self.raw["media"]

    @property
    def store(self) -> dict[str, Any]:
        return self.raw["store"]

    @property
    def live(self) -> dict[str, Any]:
        return self.raw["live"]

    @property
    def admin(self) -> dict[str, Any]:
        return self.raw["admin"]

    @property
    def pools(self) -> dict[str, Any]:
        return self.raw["pools"]

    @property
    def baseline(self) -> dict[str, Any]:
        return self.raw["baseline"]

    @property
    def environment(self) -> dict[str, Any]:
        return self.raw["environment"]


def load_spec_file(path: str | Path) -> Spec:
    p = Path(path)
    doc = json.loads(p.read_text())
    errors = validate_spec(doc)
    if errors:
        raise ValueError(f"{p}: invalid spec: " + ", ".join(errors))
    return Spec(raw=doc, path=p, fingerprint=canonical_sha256(doc))


@lru_cache(maxsize=4)
def _cached(path: str) -> Spec:
    return load_spec_file(path)


def load_spec(path: str | Path | None = None) -> Spec:
    return _cached(str(Path(path) if path else spec_path()))


def load_schema(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or schema_path()).read_text())
