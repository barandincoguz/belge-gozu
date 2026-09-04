from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

from belge_gozu.retrieval.dense import DenseModelSpec

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_dense_artifacts", REPO / "scripts" / "build_dense_artifacts.py"
)
assert _spec and _spec.loader
builder = importlib.util.module_from_spec(_spec)
sys.modules["build_dense_artifacts"] = builder
_spec.loader.exec_module(builder)


class _FakeEncoder:
    batch_size = 2

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.array([[float(len(text)), float(index + 1)] for index, text in enumerate(texts)])


def test_cli_defaults_to_single_page_batches_for_constrained_gpus() -> None:
    args = builder._parse_args(
        [
            "--index-dir",
            "index",
            "--source-repo",
            "user/index",
            "--source-revision",
            "a" * 40,
            "--model",
            "qwen3-embedding-4b",
        ]
    )

    assert args.batch_size == 1


def test_builder_writes_manifest_only_after_resumable_matrix_finishes(tmp_path: Path) -> None:
    spec = DenseModelSpec("test/model", "a" * 40, "instruction", 128)
    page_texts = {"p1": "bir", "p2": "iki"}

    result = builder.build_model_artifact(
        spec,
        page_texts,
        tmp_path,
        page_texts_sha256=hashlib.sha256(b"texts").hexdigest(),
        source_repo="user/index",
        source_revision="b" * 40,
        producer_git_commit="c" * 40,
        encoder=_FakeEncoder(),
    )

    assert (result / "embeddings.npy").is_file()
    assert (result / "dense.json").is_file()
