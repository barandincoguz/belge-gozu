import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

from belge_gozu.index.compat import check_compatibility
from belge_gozu.index.manifest import Quantization, read_manifest

_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_REPRESENTATION_FILES: dict[Quantization, tuple[str, ...]] = {
    Quantization.sign_1bit: ("tokens.npy", "page_vecs.npy"),
    Quantization.int8: ("codes.npy", "scales.npy"),
    Quantization.float16: ("embs.npy",),
}


def _api(
    api: HfApi | None,
    token: str,
    api_factory: Callable[..., HfApi],
) -> HfApi:
    return api if api is not None else api_factory(token=token or None)


def _resolved_commit(api: HfApi, repo_id: str, revision: str) -> str:
    info = api.repo_info(repo_id, repo_type="dataset", revision=revision)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not _COMMIT_SHA.fullmatch(sha):
        raise ValueError("Hugging Face geçerli bir 40 karakterli commit SHA döndürmedi")
    return sha.lower()


def push_index(
    index_dir: Path,
    repo_id: str,
    api: HfApi | None = None,
    images_dir: Path | None = None,
    *,
    token: str = "",
    revision: str = "main",
    api_factory: Callable[..., HfApi] = HfApi,
) -> str:
    """İndeksi adlandırılmış bir Hub dalına yükler ve oluşan commit SHA'yı döndürür."""
    if not repo_id:
        raise ValueError("BG_HF_DATASET_REPO ayarlı değil")
    if not revision:
        raise ValueError("yükleme revision değeri boş olamaz")
    if not index_dir.is_dir():
        raise ValueError(f"indeks dizini bulunamadı: {index_dir}")

    a = _api(api, token, api_factory)
    a.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    a.upload_folder(
        folder_path=str(index_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="index",
        revision=revision,
        # Hugging Face bu kalıbı `path_in_repo`ya göreli yorumlar.
        delete_patterns=["*"],
    )
    if images_dir is not None:
        if not images_dir.is_dir():
            raise ValueError(f"görsel dizini bulunamadı: {images_dir}")
        a.upload_folder(
            folder_path=str(images_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="images",
            revision=revision,
            delete_patterns=["*"],
        )
    return _resolved_commit(a, repo_id, revision)


def _load_page_ids(index_dir: Path) -> list[str]:
    try:
        raw = json.loads((index_dir / "page_ids.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("page_ids.json okunamadı") from exc
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise ValueError("page_ids.json bir dize listesi olmalı")
    if len(raw) != len(set(raw)):
        raise ValueError("page_ids.json yinelenen kimlik içeriyor")
    return raw


def _load_array(index_dir: Path, name: str) -> np.ndarray:
    try:
        return np.load(index_dir / name, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{name} geçerli bir NumPy dizisi değil") from exc


def _validate_staged_index(
    index_dir: Path,
    *,
    expected_model_name: str | None,
    expected_model_revision: str | None,
    expected_query_format_id: str | None,
    expected_doc_prompt_sha256: str | None,
    expected_corpus_checksum: str | None,
    require_page_texts: bool,
) -> None:
    try:
        manifest = read_manifest(index_dir)
    except (OSError, ValueError) as exc:
        raise ValueError("manifest.json okunamadı veya geçersiz") from exc
    if manifest is None:
        raise ValueError("indirilen indekste manifest.json yok")

    required = ("page_ids.json", "meta.parquet", "offsets.npy")
    missing = [name for name in required if not (index_dir / name).is_file()]
    if missing:
        raise ValueError(f"indirilen indeksin zorunlu dosyaları eksik: {', '.join(missing)}")

    try:
        quantization = Quantization(manifest.quantization)
    except ValueError as exc:
        supported = ", ".join(value.value for value in Quantization)
        raise ValueError(
            f"manifest quantization={manifest.quantization!r} tanınmıyor; desteklenen: {supported}"
        ) from exc
    representation = _REPRESENTATION_FILES[quantization]
    missing_representation = [name for name in representation if not (index_dir / name).is_file()]
    if missing_representation:
        raise ValueError(
            "manifest ile indeks temsili uyuşmuyor; eksik: " + ", ".join(missing_representation)
        )

    page_ids = _load_page_ids(index_dir)
    if len(page_ids) != manifest.n_pages:
        raise ValueError(f"manifest n_pages={manifest.n_pages}, page_ids sayısı={len(page_ids)}")

    offsets = _load_array(index_dir, "offsets.npy")
    if offsets.ndim != 1 or len(offsets) != manifest.n_pages + 1:
        raise ValueError("offsets.npy uzunluğu n_pages + 1 olmalı")
    if int(offsets[0]) != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets.npy sıfırdan başlamalı ve kesin artmalı")
    if int(offsets[-1]) != manifest.n_tokens:
        raise ValueError(f"manifest n_tokens={manifest.n_tokens}, offsets sonu={int(offsets[-1])}")

    primary = _load_array(index_dir, representation[0])
    if primary.ndim != 2 or primary.shape[0] != manifest.n_tokens:
        raise ValueError(f"{representation[0]} ilk boyutu n_tokens ile eşleşmeli")
    if quantization is Quantization.int8:
        scales = _load_array(index_dir, "scales.npy")
        if scales.ndim != 1 or scales.shape[0] != manifest.n_tokens:
            raise ValueError("scales.npy uzunluğu n_tokens ile eşleşmeli")

    try:
        meta_ids = pd.read_parquet(index_dir / "meta.parquet", columns=["page_id"])[
            "page_id"
        ].tolist()
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("meta.parquet içinde okunabilir page_id sütunu olmalı") from exc
    if meta_ids != page_ids:
        raise ValueError("meta.parquet page_id sırası page_ids.json ile eşleşmiyor")

    compatibility = check_compatibility(
        manifest,
        model_name=expected_model_name or manifest.model_name,
        model_revision=expected_model_revision,
        query_format_id=expected_query_format_id or manifest.query_format.format_id,
        doc_prompt_sha256=expected_doc_prompt_sha256,
        index_dir=index_dir,
    )
    if compatibility:
        raise ValueError("indeks uyumsuz: " + "; ".join(compatibility))
    if expected_corpus_checksum and manifest.corpus_checksum != expected_corpus_checksum:
        raise ValueError(
            "corpus_checksum beklenen sürümle uyuşmuyor: "
            f"indeks={manifest.corpus_checksum} beklenen={expected_corpus_checksum}"
        )

    page_texts = index_dir / "page_texts.parquet"
    if require_page_texts and not page_texts.is_file():
        raise ValueError("page_texts.parquet zorunlu ama indirilen indekste yok")
    if page_texts.is_file():
        try:
            text_ids = pd.read_parquet(page_texts, columns=["page_id"])["page_id"].tolist()
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("page_texts.parquet içinde okunabilir page_id sütunu olmalı") from exc
        if text_ids != page_ids:
            raise ValueError("page_texts.parquet page_id sırası page_ids.json ile eşleşmiyor")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _replace_trees_atomically(replacements: list[tuple[Path, Path]]) -> None:
    """Doğrulanmış dizinleri değiştirir; herhangi bir hatada eskileri geri koyar."""
    prepared: list[tuple[Path, Path, bool]] = []
    completed: list[tuple[Path, Path, bool]] = []
    for _, target in replacements:
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
        prepared.append((target, backup, target.exists()))

    try:
        for (staged, target), (_, backup, had_target) in zip(replacements, prepared, strict=True):
            if had_target:
                os.replace(target, backup)
            try:
                os.replace(staged, target)
            except BaseException:
                if had_target and backup.exists():
                    os.replace(backup, target)
                raise
            completed.append((target, backup, had_target))
    except BaseException:
        for target, backup, had_target in reversed(completed):
            _remove_path(target)
            if had_target and backup.exists():
                os.replace(backup, target)
        raise
    else:
        for _, backup, _ in completed:
            _remove_path(backup)


def pull_index(
    repo_id: str,
    index_dir: Path,
    api: HfApi | None = None,
    data_dir: Path | None = None,
    *,
    token: str = "",
    revision: str,
    expected_model_name: str | None = None,
    expected_model_revision: str | None = None,
    expected_query_format_id: str | None = None,
    expected_doc_prompt_sha256: str | None = None,
    expected_corpus_checksum: str | None = None,
    require_page_texts: bool = False,
    api_factory: Callable[..., HfApi] = HfApi,
) -> str:
    """Sabit bir Hub commit'ini doğrular ve mevcut indeksin yerine güvenle koyar."""
    if not repo_id:
        raise ValueError("BG_HF_DATASET_REPO ayarlı değil")
    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError("indirme revision değeri 40 karakterli commit SHA olmalı")

    a = _api(api, token, api_factory)
    resolved = _resolved_commit(a, repo_id, revision)
    if resolved != revision.lower():
        raise ValueError(
            f"istenen revision ile çözümlenen commit uyuşmuyor: {revision} != {resolved}"
        )

    index_dir = Path(index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    allow_patterns = ["index/*", "images/*"] if data_dir is not None else ["index/*"]
    with tempfile.TemporaryDirectory(
        prefix=f".{index_dir.name}.download-", dir=index_dir.parent
    ) as download_tmp:
        a.snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=download_tmp,
        )
        staged_index = Path(download_tmp) / "index"
        if not staged_index.is_dir():
            raise ValueError("Hub snapshot'ında index/ dizini yok")
        _validate_staged_index(
            staged_index,
            expected_model_name=expected_model_name,
            expected_model_revision=expected_model_revision,
            expected_query_format_id=expected_query_format_id,
            expected_doc_prompt_sha256=expected_doc_prompt_sha256,
            expected_corpus_checksum=expected_corpus_checksum,
            require_page_texts=require_page_texts,
        )

        replacements = [(staged_index, index_dir)]
        if data_dir is not None:
            images_src = Path(download_tmp) / "images"
            if not images_src.is_dir():
                raise ValueError("Hub snapshot'ında images/ dizini yok")
            images_dst = Path(data_dir) / "images"
            images_dst.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".images.download-", dir=images_dst.parent
            ) as images_tmp:
                staged_images = Path(images_tmp) / "images"
                shutil.copytree(images_src, staged_images)
                _replace_trees_atomically(replacements + [(staged_images, images_dst)])
        else:
            _replace_trees_atomically(replacements)
    return resolved
