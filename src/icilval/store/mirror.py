"""Mirror the local store to a Hugging Face dataset repo, one commit per publish."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..spec import Spec

log = logging.getLogger(__name__)


def store_files(root: Path) -> list[str]:
    """Every file of a store that belongs in the mirror, relative to its root.

    Dotfiles are the store's own bookkeeping — `.validator.lock` above all — and are not part of
    what a reader verifies, so they never leave the machine.
    """
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


class Mirror:
    def __init__(self, root: Path, repo: str, *, token: str | None = None, private: bool = False):
        from huggingface_hub import HfApi

        self.root = Path(root)
        self.repo = repo
        self.api = HfApi(token=token)
        self.api.create_repo(repo, repo_type="dataset", exist_ok=True, private=private)

    def push(self, files: list[str], message: str = "publish") -> str | None:
        from huggingface_hub import CommitOperationAdd

        ops = [
            CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(self.root / f))
            for f in sorted(set(files))
            if (self.root / f).exists()
        ]
        if not ops:
            return None
        info = self.api.create_commit(
            repo_id=self.repo, repo_type="dataset", operations=ops, commit_message=message
        )
        log.info("mirrored %d files to %s", len(ops), self.repo)
        return str(getattr(info, "oid", ""))

    def push_all(self, message: str = "full mirror") -> str:
        info = self.api.upload_folder(
            folder_path=str(self.root),
            repo_id=self.repo,
            repo_type="dataset",
            commit_message=message,
            ignore_patterns=[".*", "**/.*"],
        )
        return str(getattr(info, "oid", info))

    def replace_all(self, message: str = "replace the store") -> str:
        """One commit that makes the repo exactly the local store: every file added, every path
        the store no longer has deleted.

        `push_all` only ever adds, which is right while a store grows. It is wrong when a store is
        rebuilt — a schema change, a new pool — because the previous layout's records and clips
        would linger beside the new ones under names nothing references. Deleting and adding in a
        single commit means a reader never sees the two mixed, and never sees an empty store.
        """
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        local = store_files(self.root)
        if not local:
            raise ValueError(f"{self.root} holds no files; refusing to empty {self.repo}")
        info = self.api.repo_info(self.repo, repo_type="dataset", files_metadata=False)
        remote = {s.rfilename for s in (info.siblings or [])}
        # `.gitattributes` is the Hub's own, not ours.
        stale = sorted(remote - set(local) - {".gitattributes"})
        ops: list[Any] = [CommitOperationDelete(path_in_repo=f) for f in stale]
        ops += [
            CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(self.root / f)) for f in local
        ]
        commit = self.api.create_commit(
            repo_id=self.repo, repo_type="dataset", operations=ops, commit_message=message
        )
        log.info("replaced %s: %d files, %d stale paths removed", self.repo, len(local), len(stale))
        return str(getattr(commit, "oid", ""))


def mirror_store(
    root: str | Path,
    repo: str,
    spec: Spec,
    *,
    message: str = "publish",
    all_files: bool = False,
    prune: bool = False,
    files: list[str] | None = None,
    token: str | None = None,
) -> int:
    m = Mirror(Path(root), repo, token=token)
    if prune:
        m.replace_all(message)
        return len(store_files(Path(root)))
    if all_files or files is None:
        m.push_all(message)
        return len(store_files(Path(root)))
    m.push(files, message)
    return len(files)
