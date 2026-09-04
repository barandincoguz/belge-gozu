"""Doğrulanmış dense artefaktların değişmez Hub commit'leriyle aktarımı."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import HfApi

from belge_gozu.bench.dense_artifacts import DenseArtifactExpectation, validate_dense_artifact

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_MODEL_KEY = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _client(api: HfApi | None, token: str, factory: Callable[..., HfApi]) -> HfApi:
    return api if api is not None else factory(token=token or None)


def _require_commit_sha(revision: str) -> None:
    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError("revision 40 karakterli commit SHA olmalı")


def _require_model_key(model_key: str) -> None:
    if not _MODEL_KEY.fullmatch(model_key):
        raise ValueError("model anahtarı küçük harf, rakam ve tire içermeli")


def _resolved_commit(api: HfApi, repo_id: str, revision: str) -> str:
    info = api.repo_info(repo_id, repo_type="dataset", revision=revision)
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not _COMMIT_SHA.fullmatch(resolved):
        raise ValueError("Hugging Face geçerli bir 40 karakterli commit SHA döndürmedi")
    return resolved


def _replace_tree_atomically(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
    had_target = target.exists()
    if had_target:
        os.replace(target, backup)
    try:
        os.replace(staged, target)
    except BaseException:
        if had_target and backup.exists():
            os.replace(backup, target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def push_dense_artifact(
    artifact_dir: Path,
    repo_id: str,
    model_key: str,
    expectation: DenseArtifactExpectation,
    *,
    api: HfApi | None = None,
    token: str = "",
    revision: str = "main",
    api_factory: Callable[..., HfApi] = HfApi,
) -> str:
    """Tamamlanmış tek model dizinini Hub'a yükler ve oluşan commit'i döndürür."""
    if not repo_id:
        raise ValueError("dense artefakt Dataset repo kimliği boş olamaz")
    if not revision:
        raise ValueError("yükleme revision değeri boş olamaz")
    _require_model_key(model_key)
    validate_dense_artifact(artifact_dir, expectation)
    client = _client(api, token, api_factory)
    client.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    client.upload_folder(
        folder_path=str(artifact_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"artifacts/{model_key}",
        revision=revision,
    )
    return _resolved_commit(client, repo_id, revision)


def pull_dense_artifact(
    repo_id: str,
    revision: str,
    model_key: str,
    destination_root: Path,
    expectation: DenseArtifactExpectation,
    *,
    api: HfApi | None = None,
    token: str = "",
    api_factory: Callable[..., HfApi] = HfApi,
) -> str:
    """Sabit Hub commit'inden bir modeli doğrular ve hedefe atomik olarak kurar."""
    if not repo_id:
        raise ValueError("dense artefakt Dataset repo kimliği boş olamaz")
    _require_commit_sha(revision)
    _require_model_key(model_key)
    client = _client(api, token, api_factory)
    resolved = _resolved_commit(client, repo_id, revision)
    if resolved != revision:
        raise ValueError(
            f"istenen revision ile çözümlenen commit uyuşmuyor: {revision} != {resolved}"
        )
    destination_root = Path(destination_root)
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination_root.name}.download-", dir=destination_root.parent
    ) as download_dir:
        client.snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"artifacts/{model_key}/*"],
            local_dir=download_dir,
        )
        staged = Path(download_dir) / "artifacts" / model_key
        if not staged.is_dir():
            raise ValueError(f"Hub snapshot'ında artifacts/{model_key} dizini yok")
        validate_dense_artifact(staged, expectation)
        _replace_tree_atomically(staged, destination_root / model_key)
    return resolved
