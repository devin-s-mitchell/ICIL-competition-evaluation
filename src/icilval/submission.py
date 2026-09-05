"""Fetching a submission: a Hugging Face model repo at a pinned revision, allow-listed files only."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .ids import ModelRef
from .spec import Spec

log = logging.getLogger(__name__)


@dataclass
class FetchResult:
    path: Path
    bytes: int
    files: list[str]
    skipped: list[str]


def fetch_model(
    ref: ModelRef,
    dest: Path,
    spec: Spec,
    *,
    local_models: dict[str, str] | None = None,
    token: str | None = None,
) -> FetchResult:
    """Download allow-listed files of `ref` into `dest`. `local_models` maps repo -> directory for offline runs."""
    local = (local_models or {}).get(ref.repo)
    if local:
        p = Path(local)
        files = sorted(str(f.relative_to(p)) for f in p.rglob("*") if f.is_file())
        return FetchResult(p, sum((p / f).stat().st_size for f in files), files, [])
    from huggingface_hub import HfApi, snapshot_download

    allowed = tuple(spec.model["allowed_extensions"])
    info = HfApi(token=token).model_info(ref.repo, revision=ref.revision, files_metadata=True)
    wanted, skipped, total = [], [], 0
    for s in info.siblings or []:
        if s.rfilename.lower().endswith(allowed):
            wanted.append(s.rfilename)
            total += int(s.size or 0)
        else:
            skipped.append(s.rfilename)
    if total > int(spec.model["max_repo_bytes"]):
        raise ValueError(f"{ref.entry}: {total} bytes of allow-listed files exceeds the limit")
    if not wanted:
        raise ValueError(f"{ref.entry}: no allow-listed files")
    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        ref.repo,
        revision=ref.revision,
        local_dir=str(dest),
        allow_patterns=wanted,
        token=token,
        max_workers=4,
    )
    log.info(
        "fetched %s (%d files, %d bytes; skipped %d)", ref.entry, len(wanted), total, len(skipped)
    )
    return FetchResult(Path(path), total, wanted, skipped)
