from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from belge_gozu.bench.dataset import BenchQuestion
from belge_gozu.bench.dense_artifacts import write_dense_manifest
from belge_gozu.retrieval.expand import load_expansion_cache
from tests.bench.test_dataset import q_dict

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "eval_semantic_coverage", REPO / "scripts" / "eval_semantic_coverage.py"
)
assert _spec and _spec.loader
esc = importlib.util.module_from_spec(_spec)
sys.modules["eval_semantic_coverage"] = esc
_spec.loader.exec_module(esc)


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return np.array([[float(len(text)), float(index)] for index, text in enumerate(texts)])


def test_resumable_dense_batches_continue_from_the_saved_row(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "dense"
    identity = {"repo": "test/model", "revision": "abc", "page_ids_sha256": "123"}
    first = _FakeEncoder()

    partial = esc.resume_dense_embeddings(
        first, ["bir", "iki", "üç"], checkpoint_dir, identity, batch_size=2, max_batches=1
    )

    assert partial is None
    assert first.calls == [["bir", "iki"]]
    assert not (checkpoint_dir / "embeddings.npy").exists()
    assert esc.read_dense_progress(checkpoint_dir)["completed_rows"] == 2

    second = _FakeEncoder()
    embeddings = esc.resume_dense_embeddings(
        second, ["bir", "iki", "üç"], checkpoint_dir, identity, batch_size=2
    )

    assert second.calls == [["üç"]]
    assert embeddings is not None
    assert embeddings.shape == (3, 2)
    assert (checkpoint_dir / "embeddings.npy").exists()
    assert not (checkpoint_dir / "progress.json").exists()


def test_resumable_dense_batches_reject_a_checkpoint_for_another_input(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "dense"
    first_identity = {"repo": "test/model", "revision": "abc", "page_ids_sha256": "123"}
    esc.resume_dense_embeddings(
        _FakeEncoder(),
        ["bir", "iki", "üç"],
        checkpoint_dir,
        first_identity,
        batch_size=2,
        max_batches=1,
    )

    with pytest.raises(ValueError, match="kimliği uyuşmuyor"):
        esc.resume_dense_embeddings(
            _FakeEncoder(),
            ["bir", "iki", "üç"],
            checkpoint_dir,
            {**first_identity, "page_ids_sha256": "başka"},
            batch_size=2,
        )


def test_dense_arm_reports_not_available_without_completed_artifact(tmp_path: Path) -> None:
    arm, pages = esc._dense_arm(
        esc.DenseModelSpec("test/model", "abc", "instruction", 8),
        questions=[],
        page_ids=["p1", "p2", "p3"],
        page_texts={"p1": "bir", "p2": "iki", "p3": "üç"},
        page_texts_sha256="a" * 64,
        baseline={"bm25": {}},
        artifact_root=tmp_path,
        device="cpu",
    )

    assert arm["status"] == "not_available"
    assert pages is None
    assert not (tmp_path / "model" / "embeddings.npy").exists()


def test_verified_embeddings_reject_another_page_text_artifact(tmp_path: Path) -> None:
    spec = esc.DenseModelSpec("test/model", "a" * 40, "instruction", 128)
    artifact = tmp_path / "model"
    artifact.mkdir()
    np.save(artifact / "embeddings.npy", np.ones((1, 2), dtype=np.float32))
    write_dense_manifest(
        artifact,
        spec=spec,
        page_ids=["p1"],
        page_texts_sha256="b" * 64,
        source_repo="user/index",
        source_revision="c" * 40,
        producer_git_commit="d" * 40,
    )

    with pytest.raises(ValueError, match="page_texts"):
        esc._load_verified_embeddings(spec, ["p1"], "e" * 64, tmp_path)


class _FakeExpander:
    """Soru metnine göre hata fırlatabilen sahte genişletici."""

    def __init__(self, raises: dict[str, Exception] | None = None) -> None:
        self._raises = raises or {}
        self.calls: list[str] = []

    def preflight(self) -> None:
        return None

    def expand(self, question: str) -> str:
        self.calls.append(question)
        error = self._raises.get(question)
        if error is not None:
            raise error
        return f"{question} genisletilmis"


def _expansion_questions() -> list[BenchQuestion]:
    return [
        BenchQuestion(**q_dict(question_id="q1", question="Birinci soru?")),
        BenchQuestion(**q_dict(question_id="q2", question="Ikinci soru?")),
    ]


def _install_fake_expander(monkeypatch: pytest.MonkeyPatch, fake: _FakeExpander) -> None:
    monkeypatch.setattr(esc, "LocalQueryExpander", lambda **kwargs: fake)
    monkeypatch.setattr(esc, "_release_torch_memory", lambda: None)


def test_invalid_expansion_falls_back_to_the_original_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tek bozuk varyant bütün ölçümü düşürmez; kol koşar, kayıt raporda kalır."""
    fake = _FakeExpander({"Ikinci soru?": ValueError("genişletme özgün sorguyla aynı")})
    _install_fake_expander(monkeypatch, fake)
    cache = tmp_path / "expansion-cache.jsonl"

    expansions, invalid = esc._expansions(_expansion_questions(), cache, None)

    assert expansions["Birinci soru?"] == "Birinci soru? genisletilmis"
    assert expansions["Ikinci soru?"] == "Ikinci soru?"
    assert invalid == ["q2"]
    assert set(load_expansion_cache(cache)) == {"q1"}


def test_completed_expansions_survive_a_hard_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Üretilmiş varyantlar ANINDA diske yazılır: çöken koşum onları çöpe atmaz."""
    fake = _FakeExpander({"Ikinci soru?": RuntimeError("cihaz düştü")})
    _install_fake_expander(monkeypatch, fake)
    cache = tmp_path / "expansion-cache.jsonl"

    with pytest.raises(RuntimeError):
        esc._expansions(_expansion_questions(), cache, None)

    assert set(load_expansion_cache(cache)) == {"q1"}
