import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def _api(api: HfApi | None) -> HfApi:
    return api or HfApi()


def push_index(index_dir: Path, repo_id: str, api: HfApi | None = None) -> None:
    if not repo_id:
        raise ValueError("BG_HF_DATASET_REPO ayarlı değil")
    a = _api(api)
    a.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    a.upload_folder(
        folder_path=str(index_dir), repo_id=repo_id, repo_type="dataset", path_in_repo="index"
    )


def pull_index(repo_id: str, index_dir: Path, api: HfApi | None = None) -> None:
    if not repo_id:
        raise ValueError("BG_HF_DATASET_REPO ayarlı değil")
    a = _api(api)
    with tempfile.TemporaryDirectory() as tmp:
        a.snapshot_download(
            repo_id=repo_id, repo_type="dataset", allow_patterns=["index/*"], local_dir=tmp
        )
        src = Path(tmp) / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy(f, index_dir / f.name)
