from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from belge_gozu.retrieval.dense import DenseModelSpec


def _spec() -> DenseModelSpec:
    return DenseModelSpec("test/model", "a" * 40, "test instruction", 128)


def _page_hash(page_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(page_ids).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(path: Path, page_ids: list[str]) -> Path:
    from belge_gozu.bench.dense_artifacts import encoding_fingerprint

    path.mkdir()
    embeddings = np.arange(len(page_ids) * 3, dtype=np.float32).reshape(len(page_ids), 3) + 1
    np.save(path / "embeddings.npy", embeddings)
    page_texts_hash = hashlib.sha256(b"page-texts").hexdigest()
    (path / "dense.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"repo": _spec().repo, "revision": _spec().revision},
                "encoding_fingerprint": encoding_fingerprint(_spec()),
                "source": {"repo": "user/index", "revision": "b" * 40},
                "page_ids_sha256": _page_hash(page_ids),
                "page_texts_sha256": page_texts_hash,
                "embedding": {
                    "file": "embeddings.npy",
                    "sha256": _file_hash(path / "embeddings.npy"),
                    "dtype": "float32",
                    "shape": [len(page_ids), 3],
                },
                "producer": {"git_commit": "c" * 40},
            }
        ),
        encoding="utf-8",
    )
    return path


def _expectation(page_ids: list[str]):
    from belge_gozu.bench.dense_artifacts import DenseArtifactExpectation

    return DenseArtifactExpectation(
        model=_spec(),
        page_ids=page_ids,
        page_texts_sha256=hashlib.sha256(b"page-texts").hexdigest(),
    )


def test_validate_dense_artifact_accepts_exact_model_and_page_identity(tmp_path: Path) -> None:
    from belge_gozu.bench.dense_artifacts import validate_dense_artifact

    artifact = _write_artifact(tmp_path / "artifact", ["law/1", "law/2"])

    result = validate_dense_artifact(artifact, _expectation(["law/1", "law/2"]))

    assert result["embedding"]["shape"] == [2, 3]


def test_validate_dense_artifact_rejects_hash_mismatch_before_loading(tmp_path: Path) -> None:
    from belge_gozu.bench.dense_artifacts import validate_dense_artifact

    artifact = _write_artifact(tmp_path / "artifact", ["law/1"])
    (artifact / "embeddings.npy").write_bytes(b"altered")

    with pytest.raises(ValueError, match="SHA-256"):
        validate_dense_artifact(artifact, _expectation(["law/1"]))


def test_validate_dense_artifact_rejects_another_page_order(tmp_path: Path) -> None:
    from belge_gozu.bench.dense_artifacts import validate_dense_artifact

    artifact = _write_artifact(tmp_path / "artifact", ["law/2", "law/1"])

    with pytest.raises(ValueError, match="page_ids"):
        validate_dense_artifact(artifact, _expectation(["law/1", "law/2"]))
