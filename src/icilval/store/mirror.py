"""Mirror the local store to a Hugging Face dataset repo, one commit per publish."""

from __future__ import annotations

import logging
from pathlib import Path

from ..spec import Spec

log = logging.getLogger(__name__)


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
        )
        return str(getattr(info, "oid", info))


def mirror_store(
    root: str | Path,
    repo: str,
    spec: Spec,
    *,
    message: str = "publish",
    all_files: bool = False,
    files: list[str] | None = None,
    token: str | None = None,
) -> int:
    m = Mirror(Path(root), repo, token=token)
    if all_files or files is None:
        m.push_all(message)
        return sum(1 for _ in Path(root).rglob("*") if _.is_file())
    m.push(files, message)
    return len(files)
