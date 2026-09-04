"""Offline dense sayfa artefaktlarının taşınabilir doğrulama sözleşmesi."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from belge_gozu.retrieval.dense import DenseModelSpec

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DenseArtifactExpectation:
    """Yerel indeksle eşleşmesi zorunlu dense artefakt kimliği."""

    model: DenseModelSpec
    page_ids: Sequence[str]
    page_texts_sha256: str


def sha256_file(path: Path) -> str:
    """Bir dosyanın içeriğini sabit bloklarla SHA-256 olarak hesaplar."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def page_ids_sha256(page_ids: Sequence[str]) -> str:
    """Sıralı ve benzersiz sayfa kimliklerini doğrulanabilir biçimde özetler."""
    values = list(page_ids)
    if not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError("page_ids boş olmayan dize dizisi olmalı")
    if len(values) != len(set(values)):
        raise ValueError("page_ids benzersiz olmalı")
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def encoding_fingerprint(spec: DenseModelSpec) -> str:
    """Sayfa vektörünün model-dışı kodlama protokolünü de kimliğe katar."""
    protocol = {
        "dtype": "float32",
        "instruction": spec.instruction,
        "max_length": spec.max_length,
        "normalization": "l2",
        "pooling": "last-token",
    }
    encoded = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def dense_model_key(spec: DenseModelSpec) -> str:
    """Hub ve yerel dizin için model deposundan türetilen kararlı anahtar."""
    key = spec.repo.rsplit("/", 1)[-1].lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", key):
        raise ValueError("dense model deposundan geçerli artefakt anahtarı türetilemedi")
    return key


def _require_commit_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not _COMMIT_SHA.fullmatch(value):
        raise ValueError(f"{field} 40 karakterli küçük harf commit SHA olmalı")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} boş olmayan dize olmalı")
    return value


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("dense.json okunamadı") from exc
    if not isinstance(value, dict):
        raise ValueError("dense.json nesne olmalı")
    return value


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} nesne olmalı")
    return value


def _require_equal(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise ValueError(f"dense artefakt {field} uyuşmuyor")


def _validate_embedding(
    values: np.ndarray,
    metadata: Mapping[str, object],
    *,
    file_sha256: str,
    expected_rows: int,
) -> None:
    _require_equal(metadata.get("file"), "embeddings.npy", "embedding dosyası")
    _require_equal(metadata.get("sha256"), file_sha256, "embedding SHA-256")
    _require_equal(metadata.get("dtype"), "float32", "embedding dtype")
    _require_equal(metadata.get("shape"), list(values.shape), "embedding şekli")
    if values.ndim != 2 or values.shape[0] != expected_rows or values.shape[1] < 1:
        raise ValueError("dense artefakt embedding şekli page_ids ile uyuşmuyor")
    if values.dtype != np.dtype("float32"):
        raise ValueError("dense artefakt embedding dtype float32 olmalı")
    if not np.isfinite(values).all():
        raise ValueError("dense artefakt embedding sonlu olmalı")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def read_dense_progress(checkpoint_dir: Path) -> dict[str, object]:
    """Tamamlanmamış dense indeksin doğrulanabilir ilerleme kaydını okur."""
    path = checkpoint_dir / "progress.json"
    if not path.exists():
        raise ValueError(f"dense ilerleme kaydı yok: {path}")
    value = _read_manifest(path)
    return value


def _require_resume_identity(actual: object, expected: Mapping[str, str]) -> None:
    if not isinstance(actual, dict) or actual != dict(expected):
        raise ValueError("dense kontrol noktası kimliği uyuşmuyor")


def resume_dense_embeddings(
    encoder: Any,
    texts: Sequence[str],
    checkpoint_dir: Path,
    identity: Mapping[str, str],
    *,
    batch_size: int,
    max_batches: int | None = None,
) -> np.ndarray | None:
    """Sayfa batch'lerini güvenle sürdürür; final yalnız tüm satırlar yazılınca görünür."""
    if batch_size < 1:
        raise ValueError("dense batch_size en az 1 olmalı")
    if max_batches is not None and max_batches < 1:
        raise ValueError("dense batch bütçesi en az 1 olmalı")
    if not texts:
        raise ValueError("dense sayfa metinleri boş olamaz")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / "embeddings.npy"
    resume_path = checkpoint_dir / "resume.json"
    partial_path = checkpoint_dir / "embeddings.partial.npy"
    progress_path = checkpoint_dir / "progress.json"
    identity_dict = dict(identity)
    row_count = len(texts)
    if final_path.exists():
        if not resume_path.exists():
            raise ValueError("dense final artefakt kontrol kaydı yok")
        resume = _read_manifest(resume_path)
        _require_resume_identity({key: resume.get(key) for key in identity_dict}, identity_dict)
        values = np.load(final_path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[0] != row_count:
            raise ValueError("dense final satır sayısı uyuşmuyor")
        return np.asarray(values)

    completed_rows = 0
    matrix: np.memmap | None = None
    if progress_path.exists() or partial_path.exists():
        if not progress_path.exists() or not partial_path.exists():
            raise ValueError("dense kontrol noktası eksik dosya içeriyor")
        progress = read_dense_progress(checkpoint_dir)
        _require_resume_identity(progress.get("identity"), identity_dict)
        completed_value = progress.get("completed_rows")
        dimension_value = progress.get("dimension")
        if progress.get("row_count") != row_count or not isinstance(completed_value, int):
            raise ValueError("dense kontrol noktası satır sayısı uyuşmuyor")
        if not isinstance(dimension_value, int) or not 0 <= completed_value < row_count:
            raise ValueError("dense kontrol noktası ilerlemesi geçersiz")
        completed_rows = completed_value
        matrix = np.lib.format.open_memmap(partial_path, mode="r+")
        if matrix.shape != (row_count, dimension_value):
            raise ValueError("dense kontrol noktası embedding şekli uyuşmuyor")

    batches_done = 0
    started = time.perf_counter()
    for start in range(completed_rows, row_count, batch_size):
        if max_batches is not None and batches_done >= max_batches:
            return None
        end = min(start + batch_size, row_count)
        batch = np.asarray(encoder.encode_passages(list(texts[start:end])), dtype=np.float32)
        if batch.ndim != 2 or batch.shape[0] != end - start or batch.shape[1] < 1:
            raise ValueError("dense batch embedding şekli geçersiz")
        if matrix is None:
            matrix = np.lib.format.open_memmap(
                partial_path, mode="w+", dtype=np.float32, shape=(row_count, batch.shape[1])
            )
        if batch.shape[1] != matrix.shape[1]:
            raise ValueError("dense batch embedding boyutu değişti")
        matrix[start:end] = batch
        matrix.flush()
        completed_rows = end
        batches_done += 1
        _atomic_json(
            progress_path,
            {
                "identity": identity_dict,
                "row_count": row_count,
                "dimension": int(matrix.shape[1]),
                "completed_rows": completed_rows,
            },
        )
        elapsed = time.perf_counter() - started
        print(
            f"dense progress {checkpoint_dir.name}: {completed_rows}/{row_count}; "
            f"elapsed={elapsed:.1f}s"
        )

    assert matrix is not None
    matrix.flush()
    _atomic_json(
        resume_path,
        {**identity_dict, "row_count": row_count, "dimension": int(matrix.shape[1])},
    )
    del matrix
    os.replace(partial_path, final_path)
    progress_path.unlink()
    return np.load(final_path)


def write_dense_manifest(
    artifact_dir: Path,
    *,
    spec: DenseModelSpec,
    page_ids: Sequence[str],
    page_texts_sha256: str,
    source_repo: str,
    source_revision: str,
    producer_git_commit: str,
) -> dict[str, Any]:
    """Tamamlanmış matrisi açıklayan doğrulanabilir dense.json dosyasını yazar."""
    artifact_dir = Path(artifact_dir)
    embeddings_path = artifact_dir / "embeddings.npy"
    if not embeddings_path.is_file():
        raise ValueError("tamamlanmış embeddings.npy bulunamadı")
    if (artifact_dir / "embeddings.partial.npy").exists() or (
        artifact_dir / "progress.json"
    ).exists():
        raise ValueError("tamamlanmamış dense kontrol noktası manifestlenemez")
    _require_string(source_repo, "source repo")
    _require_commit_sha(source_revision, "source revision")
    _require_commit_sha(producer_git_commit, "producer git commit")
    _require_string(page_texts_sha256, "page_texts_sha256")
    values = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    ids_hash = page_ids_sha256(page_ids)
    if values.ndim != 2 or values.shape[0] != len(page_ids) or values.shape[1] < 1:
        raise ValueError("embeddings.npy page_ids ile hizalı iki boyutlu olmalı")
    if values.dtype != np.dtype("float32") or not np.isfinite(values).all():
        raise ValueError("embeddings.npy sonlu float32 olmalı")
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "model": {"repo": spec.repo, "revision": spec.revision},
        "encoding_fingerprint": encoding_fingerprint(spec),
        "source": {"repo": source_repo, "revision": source_revision},
        "page_ids_sha256": ids_hash,
        "page_texts_sha256": page_texts_sha256,
        "embedding": {
            "file": "embeddings.npy",
            "sha256": sha256_file(embeddings_path),
            "dtype": "float32",
            "shape": list(values.shape),
        },
        "producer": {"git_commit": producer_git_commit},
    }
    (artifact_dir / "dense.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_dense_artifact(
    artifact_dir: Path, expectation: DenseArtifactExpectation
) -> dict[str, Any]:
    """Bir dense artefaktı yüklemeden önce kimlik ve dosya bütünlüğünü doğrular."""
    artifact_dir = Path(artifact_dir)
    manifest = _read_manifest(artifact_dir / "dense.json")
    _require_equal(manifest.get("schema_version"), _SCHEMA_VERSION, "şema sürümü")
    _require_equal(
        manifest.get("model"),
        {"repo": expectation.model.repo, "revision": expectation.model.revision},
        "model",
    )
    _require_equal(
        manifest.get("encoding_fingerprint"),
        encoding_fingerprint(expectation.model),
        "encoding fingerprint",
    )
    _require_equal(
        manifest.get("page_ids_sha256"), page_ids_sha256(expectation.page_ids), "page_ids"
    )
    _require_equal(manifest.get("page_texts_sha256"), expectation.page_texts_sha256, "page_texts")
    source = _require_mapping(manifest.get("source"), "source")
    _require_string(source.get("repo"), "source repo")
    _require_commit_sha(source.get("revision"), "source revision")
    producer = _require_mapping(manifest.get("producer"), "producer")
    _require_commit_sha(producer.get("git_commit"), "producer git commit")
    metadata = _require_mapping(manifest.get("embedding"), "embedding")
    embeddings_path = artifact_dir / "embeddings.npy"
    if not embeddings_path.is_file():
        raise ValueError("dense artefakt embeddings.npy yok")
    _require_equal(metadata.get("sha256"), sha256_file(embeddings_path), "embedding SHA-256")
    try:
        values = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("dense artefakt embeddings.npy okunamadı") from exc
    _validate_embedding(
        values,
        metadata,
        file_sha256=sha256_file(embeddings_path),
        expected_rows=len(expectation.page_ids),
    )
    return manifest
