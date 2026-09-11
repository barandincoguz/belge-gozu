# ruff: noqa: E402

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from eval_candidate_reranker import (  # pyright: ignore[reportMissingImports]
    _DenseCandidateChannel,
    _MaxPScorer,
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
    assert (
        report["candidate_pool"]["coverage"]["overall"]
        >= report["pinned"]["overall"]["recall_at"][50]
    )


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


class TwoQueryText:
    """Özgün ve genişletilmiş sorguya FARKLI skor veren metin kanalı."""

    page_ids = ["b1", "a1", "a2", "x9"]

    def scores(self, query: str) -> np.ndarray:
        if query == "soru":
            return np.array([12.0, 7.0, 11.0, 0.0])
        assert query == "kanun dilinde soru"
        return np.array([0.0, 1.0, 2.0, 9.0])


class TwoQueryLate:
    """Her iki sorguyu da kabul eder; geç kanal İKİ kez sorgulanır."""

    def __init__(self, pages: list[str]) -> None:
        self.pages = pages
        self.queries: list[str] = []

    def candidate_pages(self, query: str, limit: int) -> list[str]:
        self.queries.append(query)
        return self.pages[:limit]


class OriginalOnlyReranker:
    """Yeniden sıralamanın ÖZGÜN soruyla yapıldığını sözleşme olarak kilitler."""

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        assert query == "soru"
        return np.array([float(len(documents) - index) for index in range(len(documents))])


def _run_with(query_for_channels: dict[str, str] | None, late: TwoQueryLate | None = None) -> dict:
    late = late or TwoQueryLate(["a1"])
    return run_comparison(
        questions=[Question(gold_page_ids=["x9"])],
        text=TwoQueryText(),
        doc_names={},
        page_texts={"b1": "bm25", "a1": "orta", "a2": "iyi", "x9": "kanun dili"},
        late_channels=[late],
        reranker=OriginalOnlyReranker(),
        candidate_limit=2,
        query_for_channels=query_for_channels,
    )


def test_expansion_only_pages_are_absent_without_the_expanded_query():
    pool = _run_with(None)["unpinned"]["diagnostics"][0]["candidate_pool"]

    assert "x9" not in pool


def test_expanded_query_feeds_the_pool_while_reranking_stays_on_the_original():
    late = TwoQueryLate(["a1"])
    report = _run_with({"q1": "kanun dilinde soru"}, late)

    assert late.queries == ["soru", "kanun dilinde soru"]
    pool = report["unpinned"]["diagnostics"][0]["candidate_pool"]
    assert "x9" in pool
    assert pool[:3] == ["b1", "a2", "a1"], "özgün sorgunun adayları ÖNCE gelir"
    assert report["pinned"]["overall"]["recall_at"][5] == 1.0


def test_diagnostics_carry_the_gold_rank_in_every_ranking():
    """Sıralama arızası teşhisi için gold'un KAÇINCI sırada olduğu raporlanır."""
    report = _run_with(None)

    ranks = report["unpinned"]["diagnostics"][0]["gold_rank"]
    assert ranks["pool"] is None, "x9 havuza girmiyor"
    assert ranks["pinned"] is None and ranks["unpinned"] is None

    expanded = _run_with({"q1": "kanun dilinde soru"})["unpinned"]["diagnostics"][0]["gold_rank"]
    assert expanded["pool"] == 4
    # sahte reranker havuz sırasını koruyor: gold yeniden sıralamada da 4'üncü
    assert expanded["pinned"] == 4


class CountingReranker:
    """Skorladığı belgeleri kaydeder; MaxP'nin chunk gönderdiğini kanıtlar."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen: list[str] = []

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        self.seen.extend(documents)
        return np.array([self.scores[document] for document in documents])


def test_maxp_scores_chunks_and_takes_the_page_maximum():
    """Sayfa skoru chunk'ların MAKSİMUMU; chunk'sız sayfa sayfa metnine düşer."""
    reranker = CountingReranker({"m1": 0.1, "m2": 0.9, "tablo sayfası": 0.5})
    scorer = _MaxPScorer(
        reranker,
        {"a1": ["m1", "m2"], "b1": []},
        {"a1": "a1 sayfa metni", "b1": "tablo sayfası"},
    )

    scores = scorer("soru", ["a1", "b1"])

    assert scores.tolist() == [0.9, 0.5]
    assert reranker.seen == ["m1", "m2", "tablo sayfası"], "sayfa metni DEĞİL chunk skorlanır"
