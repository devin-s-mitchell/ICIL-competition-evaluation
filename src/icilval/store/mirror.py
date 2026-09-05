"""Mirror the local store to a Hugging Face dataset repo (filled in with the daemon)."""

from __future__ import annotations

from pathlib import Path

from ..spec import Spec


def mirror_store(
    root: str | Path,
    repo: str,
    spec: Spec,
    *,
    message: str = "publish",
    all_files: bool = False,
    files: list[str] | None = None,
) -> int:
    raise NotImplementedError("mirror is implemented in step 4")
