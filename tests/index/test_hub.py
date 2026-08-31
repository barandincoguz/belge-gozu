import hashlib
import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import belge_gozu.index.hub as hub_module
from belge_gozu.index.hub import pull_index, push_index
from belge_gozu.index.manifest import (
    TRAIN_COMPAT_DOC_PROMPT,
    TRAIN_COMPAT_V1,
    IndexManifest,
    RenderConfig,
    corpus_checksum,
    write_manifest,
)

REVISION = "a" * 40


def _make_index(path: Path, page_ids: list[str] | None = None) -> Path:
    page_ids = page_ids or ["doc/0001", "doc/0002"]
    path.mkdir(parents=True)
    (path / "page_ids.json").write_text(json.dumps(page_ids), encoding="utf-8")
    pd.DataFrame(
        {
            "page_id": page_ids,
            "doc_name": ["doc.pdf"] * len(page_ids),
            "page_no": list(range(1, len(page_ids) + 1)),
        }
    ).to_parquet(path / "meta.parquet", index=False)
    np.save(path / "offsets.npy", np.arange(len(page_ids) + 1, dtype=np.int64))
    np.save(path / "codes.npy", np.ones((len(page_ids), 2), dtype=np.int8))
    np.save(path / "scales.npy", np.ones(2, dtype=np.float32))
    pd.DataFrame({"page_id": page_ids, "text": ["metin"] * len(page_ids)}).to_parquet(
        path / "page_texts.parquet", index=False
    )
    write_manifest(
        path,
        IndexManifest(
            model_name="vidore/colSmol-500M",
            model_revision="model-sha",
            engine_versions={"colpali-engine": "test"},
            query_format=TRAIN_COMPAT_V1,
            doc_prompt_sha256=hashlib.sha256(TRAIN_COMPAT_DOC_PROMPT.encode()).hexdigest(),
            quantization="int8",
            mask_policy="drop-padding",
            render=RenderConfig(),
            corpus_checksum=corpus_checksum(path),
            n_pages=len(page_ids),
            n_tokens=len(page_ids),
            built_at="2026-08-31T00:00:00Z",
            git_commit="test",
        ),
    )
    return path


def _expected() -> dict[str, object]:
    return {
        "expected_model_name": "vidore/colSmol-500M",
        "expected_model_revision": None,
        "expected_query_format_id": TRAIN_COMPAT_V1.format_id,
        "expected_doc_prompt_sha256": hashlib.sha256(TRAIN_COMPAT_DOC_PROMPT.encode()).hexdigest(),
        "require_page_texts": True,
    }


def _api_for(source: Path, images: Path | None = None) -> MagicMock:
    api = MagicMock()
    api.repo_info.return_value.sha = REVISION

    def fake_snapshot(**kwargs):
        local = Path(kwargs["local_dir"])
        shutil.copytree(source, local / "index")
        if images is not None:
            shutil.copytree(images, local / "images")
        return str(local)

    api.snapshot_download.side_effect = fake_snapshot
    return api


def test_push_uses_token_revision_and_returns_commit_sha(tmp_path: Path):
    index_dir = _make_index(tmp_path / "source")
    api = MagicMock()
    api.repo_info.return_value.sha = REVISION
    api_factory = MagicMock(return_value=api)

    result = push_index(
        index_dir,
        "user/belge-gozu-index",
        token="secret-token",
        revision="main",
        api_factory=api_factory,
    )

    assert result == REVISION
    api_factory.assert_called_once_with(token="secret-token")
    api.create_repo.assert_called_once_with(
        "user/belge-gozu-index", repo_type="dataset", exist_ok=True
    )
    upload = api.upload_folder.call_args.kwargs
    assert upload["revision"] == "main"
    assert upload["delete_patterns"] == ["*"]
    api.repo_info.assert_called_once_with(
        "user/belge-gozu-index", repo_type="dataset", revision="main"
    )


def test_push_uploads_images_on_same_revision(tmp_path: Path):
    index_dir = _make_index(tmp_path / "source")
    images_dir = tmp_path / "images"
    (images_dir / "doc").mkdir(parents=True)
    (images_dir / "doc" / "0001.webp").write_bytes(b"img")
    api = MagicMock()
    api.repo_info.return_value.sha = REVISION

    push_index(
        index_dir,
        "user/belge-gozu-index",
        api=api,
        images_dir=images_dir,
        revision="release",
    )

    assert api.upload_folder.call_count == 2
    index_upload, image_upload = api.upload_folder.call_args_list
    assert index_upload.kwargs["path_in_repo"] == "index"
    assert index_upload.kwargs["delete_patterns"] == ["*"]
    assert image_upload.kwargs["path_in_repo"] == "images"
    assert image_upload.kwargs["delete_patterns"] == ["*"]
    assert index_upload.kwargs["revision"] == image_upload.kwargs["revision"] == "release"


@pytest.mark.parametrize("repo_id,revision", [("", "main"), ("user/repo", "")])
def test_push_rejects_empty_repo_or_revision(tmp_path: Path, repo_id: str, revision: str):
    with pytest.raises(ValueError):
        push_index(tmp_path, repo_id, revision=revision)


def test_pull_uses_token_and_exact_revision(tmp_path: Path):
    source = _make_index(tmp_path / "source")
    api = _api_for(source)
    api_factory = MagicMock(return_value=api)

    result = pull_index(
        "user/belge-gozu-index",
        tmp_path / "target",
        token="secret-token",
        revision=REVISION,
        api_factory=api_factory,
        **_expected(),
    )

    assert result == REVISION
    api_factory.assert_called_once_with(token="secret-token")
    api.repo_info.assert_called_once_with(
        "user/belge-gozu-index", repo_type="dataset", revision=REVISION
    )
    snapshot = api.snapshot_download.call_args.kwargs
    assert snapshot["revision"] == REVISION
    assert snapshot["allow_patterns"] == ["index/*"]


@pytest.mark.parametrize("revision", ["", "main", "abc123"])
def test_pull_rejects_non_immutable_revision(tmp_path: Path, revision: str):
    with pytest.raises(ValueError, match="40 karakterli commit SHA"):
        pull_index("user/repo", tmp_path / "target", revision=revision)


def test_pull_rejects_resolved_sha_mismatch_without_touching_target(tmp_path: Path):
    source = _make_index(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    (target / "old.marker").write_text("old")
    api = _api_for(source)
    api.repo_info.return_value.sha = "b" * 40

    with pytest.raises(ValueError, match="çözümlenen commit"):
        pull_index("user/repo", target, api=api, revision=REVISION, **_expected())

    assert (target / "old.marker").read_text() == "old"


def test_interrupted_download_leaves_target_unchanged(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "old.marker").write_text("old")
    api = MagicMock()
    api.repo_info.return_value.sha = REVISION
    api.snapshot_download.side_effect = RuntimeError("network interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        pull_index("user/repo", target, api=api, revision=REVISION)

    assert (target / "old.marker").read_text() == "old"


def test_missing_manifest_leaves_target_unchanged(tmp_path: Path):
    source = _make_index(tmp_path / "source")
    (source / "manifest.json").unlink()
    target = tmp_path / "target"
    target.mkdir()
    (target / "old.marker").write_text("old")

    with pytest.raises(ValueError, match="manifest.json"):
        pull_index("user/repo", target, api=_api_for(source), revision=REVISION)

    assert (target / "old.marker").read_text() == "old"


def test_incompatible_checksum_leaves_target_unchanged(tmp_path: Path):
    source = _make_index(tmp_path / "source")
    meta = pd.read_parquet(source / "meta.parquet")
    meta.loc[0, "doc_name"] = "corrupt.pdf"
    meta.to_parquet(source / "meta.parquet", index=False)
    target = tmp_path / "target"
    target.mkdir()
    (target / "old.marker").write_text("old")

    with pytest.raises(ValueError, match="corpus_checksum"):
        pull_index("user/repo", target, api=_api_for(source), revision=REVISION)

    assert (target / "old.marker").read_text() == "old"


def test_success_replaces_tree_and_removes_stale_representation(tmp_path: Path):
    source = _make_index(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    (target / "tokens.npy").write_bytes(b"stale")

    pull_index("user/repo", target, api=_api_for(source), revision=REVISION, **_expected())

    assert (target / "codes.npy").exists()
    assert not (target / "tokens.npy").exists()
    assert not list(tmp_path.glob(".target.backup-*"))


def test_second_replace_failure_restores_previous_tree(tmp_path: Path, monkeypatch):
    source = _make_index(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    (target / "old.marker").write_text("old")
    real_replace = os.replace
    calls = 0

    def fail_second_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("swap failed")
        return real_replace(src, dst)

    monkeypatch.setattr(hub_module.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="swap failed"):
        pull_index("user/repo", target, api=_api_for(source), revision=REVISION, **_expected())

    assert (target / "old.marker").read_text() == "old"
    assert not list(tmp_path.glob(".target.backup-*"))


def test_page_text_ids_must_exactly_match_page_ids(tmp_path: Path):
    source = _make_index(tmp_path / "source")
    texts = pd.read_parquet(source / "page_texts.parquet")
    texts["page_id"] = list(reversed(texts["page_id"].tolist()))
    texts.to_parquet(source / "page_texts.parquet", index=False)
    target = tmp_path / "target"
    target.mkdir()
    (target / "old.marker").write_text("old")

    with pytest.raises(ValueError, match="page_texts.*page_ids"):
        pull_index(
            "user/repo",
            target,
            api=_api_for(source),
            revision=REVISION,
            **_expected(),
        )

    assert (target / "old.marker").read_text() == "old"


def test_pull_replaces_images_tree_without_leaving_stale_files(tmp_path: Path):
    source = _make_index(tmp_path / "source")
    images = tmp_path / "remote-images"
    (images / "doc").mkdir(parents=True)
    (images / "doc" / "0001.webp").write_bytes(b"new")
    data_dir = tmp_path / "data"
    (data_dir / "images" / "old").mkdir(parents=True)
    (data_dir / "images" / "old" / "stale.webp").write_bytes(b"old")
    api = _api_for(source, images)

    pull_index(
        "user/repo",
        tmp_path / "target",
        api=api,
        data_dir=data_dir,
        revision=REVISION,
        **_expected(),
    )

    assert (data_dir / "images" / "doc" / "0001.webp").read_bytes() == b"new"
    assert not (data_dir / "images" / "old").exists()
    assert api.snapshot_download.call_args.kwargs["allow_patterns"] == [
        "index/*",
        "images/*",
    ]
