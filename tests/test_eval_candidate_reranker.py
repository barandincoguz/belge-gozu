# ruff: noqa: E402

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from eval_candidate_reranker import (  # pyright: ignore[reportMissingImports]
    _DenseCandidateChannel,
    run_comparison,
)

from belge_gozu.retrieval.dense import DensePageIndex


@dataclass(frozen=True)
class Question:
    question_id: str = "q1"
    question: str = "soru"
    answerable: bool = True
    gold_page_ids: list[str] | None = None
    slice: str = "paraphrase"

    def __post_init__(self) -> None:
        if self.gold_page_ids is None:
            object.__setattr__(self, "gold_page_ids", ["a1"])


class FixedText:
    page_ids = ["b1", "a1", "a2"]

    def scores(self, query: str) -> np.ndarray:
        assert query == "soru"
        return np.array([12.0, 7.0, 11.0])


class FixedLate:
    def __init__(self, pages: list[str]) -> None:
        self.pages = pages

    def candidate_pages(self, query: str, limit: int) -> list[str]:
        assert query == "soru"
        return self.pages[:limit]


class FixedReranker:
    def score(self, query: str, documents: list[str]) -> np.ndarray:
        assert query == "soru"
        assert documents == ["bm25", "orta", "en iyi"]
        return np.array([0.1, 0.5, 0.9])


def test_run_comparison_reports_both_arms_and_bm25_top1_rank():
    report = run_comparison(
        questions=[Question()],
        text=FixedText(),
        doc_names={},
        page_texts={"b1": "bm25", "a1": "en iyi", "a2": "orta"},
        late_channels=[FixedLate(["a1"]), FixedLate(["a2"])],
        reranker=FixedReranker(),
    )

    assert report["pinned"]["overall"]["recall_at"][5] == 1.0
    assert report["unpinned"]["diagnostics"][0]["bm25_top1_rank_unpinned"] == 3
    assert report["candidate_pool"]["coverage"]["overall"] >= report["pinned"]["overall"][
        "recall_at"
    ][50]


class FakeDenseEncoder:
    """Sorguyu sabit bir yöne kodlar; dense sıralamasını deterministik yapar."""

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        assert len(texts) == 1
        return np.array([[1.0, 0.0]], dtype=np.float32)


def _dense_channel() -> _DenseCandidateChannel:
    index = DensePageIndex(
        ["a1", "b1", "a2"],
        np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]], dtype=np.float32),
    )
    return _DenseCandidateChannel(index, FakeDenseEncoder())


def test_dense_channel_returns_the_dense_index_top_pages():
    assert _dense_channel().candidate_pages("soru", 2) == ["b1", "a2"]


def test_dense_channel_pages_enter_the_reranked_pool():
    """Dense kanalı ekli koşumda havuz dense adaylarını da taşır."""
    report = run_comparison(
        questions=[Question()],
        text=FixedText(),
        doc_names={},
        page_texts={"b1": "bm25", "a1": "en iyi", "a2": "orta"},
        late_channels=[FixedLate(["a1"]), _dense_channel()],
        reranker=FixedReranker(),
    )

    pool = report["unpinned"]["diagnostics"][0]["candidate_pool"]
    assert set(pool) == {"b1", "a1", "a2"}
    assert report["pinned"]["overall"]["recall_at"][5] == 1.0
