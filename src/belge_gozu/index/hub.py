import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def _api(api: HfApi | None) -> HfApi:
    return api or HfApi()


def push_index(
    index_dir: Path, repo_id: str, api: HfApi | None = None, images_dir: Path | None = None
) -> None:
    if not repo_id:
        raise ValueError("BG_HF_DATASET_REPO ayarlı değil")
    a = _api(api)
    a.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    a.upload_folder(
        folder_path=str(index_dir), repo_id=repo_id, repo_type="dataset", path_in_repo="index"
    )
    if images_dir is not None:
        a.upload_folder(
            folder_path=str(images_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="images",
        )


def pull_index(
    repo_id: str, index_dir: Path, api: HfApi | None = None, data_dir: Path | None = None
) -> None:
    if not repo_id:
        raise ValueError("BG_HF_DATASET_REPO ayarlı değil")
    a = _api(api)
    allow_patterns = ["index/*", "images/*"] if data_dir is not None else ["index/*"]
    with tempfile.TemporaryDirectory() as tmp:
        a.snapshot_download(
            repo_id=repo_id, repo_type="dataset", allow_patterns=allow_patterns, local_dir=tmp
        )
        src = Path(tmp) / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy(f, index_dir / f.name)
        if data_dir is not None:
            images_src = Path(tmp) / "images"
            if images_src.exists():
                images_dst = data_dir / "images"
                images_dst.mkdir(parents=True, exist_ok=True)
                shutil.copytree(images_src, images_dst, dirs_exist_ok=True)
