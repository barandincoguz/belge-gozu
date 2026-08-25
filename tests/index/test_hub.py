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
