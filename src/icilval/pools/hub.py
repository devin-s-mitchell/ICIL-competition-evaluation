"""Pool distribution through a Hugging Face dataset repo: pools/<version>/…"""

from __future__ import annotations

from pathlib import Path


def push_pool(root: Path, repo: str, version: str, *, token: str | None = None) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    info = api.upload_folder(
        folder_path=str(root),
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=f"pools/{version}",
        commit_message=f"pool {version}",
    )
    return str(getattr(info, "oid", info))


def pull_pool(
    repo: str, version: str, dest: Path, *, revision: str | None = None, token: str | None = None
) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"pools/{version}/*", f"pools/{version}/**"],
        local_dir=str(dest),
        token=token,
    )
    return Path(path) / "pools" / version
