from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from belge_gozu.bench.dense_artifacts import (
    DenseArtifactExpectation,
    write_dense_manifest,
)
from belge_gozu.retrieval.dense import DenseModelSpec

REVISION = "a" * 40
MODEL_KEY = "qwen3-embedding-4b"


def _spec() -> DenseModelSpec:
    return DenseModelSpec("test/model", "b" * 40, "instruction", 128)


def _expectation() -> DenseArtifactExpectation:
    return DenseArtifactExpectation(
        model=_spec(),
        page_ids=["law/1", "law/2"],
        page_texts_sha256=hashlib.sha256(b"texts").hexdigest(),
    )


def _valid_artifact(path: Path) -> Path:
    path.mkdir(parents=True)
    np.save(path / "embeddings.npy", np.ones((2, 3), dtype=np.float32))
    write_dense_manifest(
        path,
        spec=_spec(),
        page_ids=["law/1", "law/2"],
        page_texts_sha256=hashlib.sha256(b"texts").hexdigest(),
        source_repo="user/index",
        source_revision="c" * 40,
        producer_git_commit="d" * 40,
    )
    return path


def _api_copying(source: Path) -> MagicMock:
    api = MagicMock()
    api.repo_info.return_value.sha = REVISION

    def snapshot_download(**kwargs: object) -> str:
        local = Path(str(kwargs["local_dir"]))
        destination = local / "artifacts" / MODEL_KEY
        destination.parent.mkdir(parents=True)
        shutil.copytree(source, destination)
        return str(local)

    api.snapshot_download.side_effect = snapshot_download
    return api


def test_pull_rejects_a_branch_without_touching_existing_model(tmp_path: Path) -> None:
    from belge_gozu.bench.dense_artifact_hub import pull_dense_artifact

    destination = tmp_path / "dense"
    target = destination / MODEL_KEY
    target.mkdir(parents=True)
    (target / "old.marker").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="40 karakterli commit SHA"):
        pull_dense_artifact("user/repo", "main", MODEL_KEY, destination, _expectation())

    assert (target / "old.marker").read_text(encoding="utf-8") == "old"


def test_pull_validates_staged_artifact_before_atomic_replacement(tmp_path: Path) -> None:
    from belge_gozu.bench.dense_artifact_hub import pull_dense_artifact

    source = _valid_artifact(tmp_path / "source")
    (source / "embeddings.npy").write_bytes(b"altered")
    destination = tmp_path / "dense"
    target = destination / MODEL_KEY
    target.mkdir(parents=True)
    (target / "old.marker").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        pull_dense_artifact(
            "user/repo", REVISION, MODEL_KEY, destination, _expectation(), api=_api_copying(source)
        )

    assert (target / "old.marker").read_text(encoding="utf-8") == "old"


def test_push_uploads_only_completed_model_directory_and_returns_resolved_sha(
    tmp_path: Path,
) -> None:
    from belge_gozu.bench.dense_artifact_hub import push_dense_artifact

    api = MagicMock()
    api.repo_info.return_value.sha = REVISION

    result = push_dense_artifact(
        _valid_artifact(tmp_path / "source"),
        "user/repo",
        MODEL_KEY,
        _expectation(),
        api=api,
    )

    assert result == REVISION
    assert api.upload_folder.call_args.kwargs["path_in_repo"] == f"artifacts/{MODEL_KEY}"
    assert "delete_patterns" not in api.upload_folder.call_args.kwargs
