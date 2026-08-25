from pathlib import Path
from unittest.mock import MagicMock

import pytest

from belge_gozu.index.hub import pull_index, push_index


def test_push_calls_hub(tmp_path: Path):
    (tmp_path / "tokens.npy").write_bytes(b"x")
    api = MagicMock()
    push_index(tmp_path, "user/belge-gozu-index", api=api)
    api.create_repo.assert_called_once_with(
        "user/belge-gozu-index", repo_type="dataset", exist_ok=True
    )
    api.upload_folder.assert_called_once()
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_type"] == "dataset" and kwargs["path_in_repo"] == "index"


def test_push_empty_repo_id_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        push_index(tmp_path, "")


def test_push_uploads_images_when_images_dir_given(tmp_path: Path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "tokens.npy").write_bytes(b"x")
    images_dir = tmp_path / "images"
    (images_dir / "k1").mkdir(parents=True)
    (images_dir / "k1" / "0001.webp").write_bytes(b"img")
    api = MagicMock()

    push_index(index_dir, "user/belge-gozu-index", api=api, images_dir=images_dir)

    assert api.upload_folder.call_count == 2
    first_kwargs = api.upload_folder.call_args_list[0].kwargs
    assert first_kwargs["folder_path"] == str(index_dir)
    assert first_kwargs["path_in_repo"] == "index"
    second_kwargs = api.upload_folder.call_args_list[1].kwargs
    assert second_kwargs["folder_path"] == str(images_dir)
    assert second_kwargs["repo_type"] == "dataset"
    assert second_kwargs["path_in_repo"] == "images"


def test_push_without_images_dir_uploads_only_index(tmp_path: Path):
    (tmp_path / "tokens.npy").write_bytes(b"x")
    api = MagicMock()
    push_index(tmp_path, "user/belge-gozu-index", api=api, images_dir=None)
    assert api.upload_folder.call_count == 1


def test_pull_moves_files(tmp_path: Path):
    api = MagicMock()

    def fake_snapshot(**kwargs):
        d = Path(kwargs["local_dir"]) / "index"
        d.mkdir(parents=True, exist_ok=True)
        (d / "tokens.npy").write_bytes(b"x")
        return str(kwargs["local_dir"])

    api.snapshot_download.side_effect = fake_snapshot
    out = tmp_path / "idx"
    pull_index("user/belge-gozu-index", out, api=api)
    assert (out / "tokens.npy").exists()
    kwargs = api.snapshot_download.call_args.kwargs
    assert kwargs["allow_patterns"] == ["index/*"]


def test_pull_moves_index_and_images_when_data_dir_given(tmp_path: Path):
    api = MagicMock()

    def fake_snapshot(**kwargs):
        d = Path(kwargs["local_dir"]) / "index"
        d.mkdir(parents=True, exist_ok=True)
        (d / "tokens.npy").write_bytes(b"x")
        img_d = Path(kwargs["local_dir"]) / "images" / "k1"
        img_d.mkdir(parents=True, exist_ok=True)
        (img_d / "0001.webp").write_bytes(b"img")
        return str(kwargs["local_dir"])

    api.snapshot_download.side_effect = fake_snapshot
    out = tmp_path / "idx"
    data_dir = tmp_path / "data"
    pull_index("user/belge-gozu-index", out, api=api, data_dir=data_dir)

    assert (out / "tokens.npy").exists()
    assert (data_dir / "images" / "k1" / "0001.webp").exists()
    kwargs = api.snapshot_download.call_args.kwargs
    assert kwargs["allow_patterns"] == ["index/*", "images/*"]
