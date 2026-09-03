import numpy as np
import pytest

from belge_gozu.retrieval.rerank import TransformerPageReranker, compare_rerankings


class FixedReranker:
    def score(self, query: str, documents: list[str]) -> np.ndarray:
        assert query == "soru"
        assert documents == ["bm25", "en iyi", "orta"]
        return np.array([0.1, 0.9, 0.5])


def test_comparison_pins_bm25_first_but_unpinned_exposes_its_rank():
    result = compare_rerankings(
        "soru",
        ["b1", "a1", "a2"],
        {"b1": "bm25", "a1": "en iyi", "a2": "orta"},
        {"b1": 11.0, "a1": 7.0, "a2": 12.0},
        FixedReranker(),
        threshold=10.6,
    )
    assert result.pinned_pages == ("b1", "a1", "a2")
    assert result.unpinned_pages == ("a1", "a2", "b1")
    assert result.bm25_top1_rank_unpinned == 3
    assert result.unpinned_top1_bm25_score == 7.0
    assert result.would_abstain is True


def test_comparison_rejects_missing_page_text():
    with pytest.raises(ValueError, match="a1"):
        compare_rerankings(
            "soru", ["b1", "a1"], {"b1": "var"}, {"b1": 11.0, "a1": 12.0}, FixedReranker()
        )


def test_comparison_rejects_wrong_length_and_non_finite_scores():
    class BadReranker:
        def __init__(self, scores: np.ndarray) -> None:
            self.scores = scores

        def score(self, query: str, documents: list[str]) -> np.ndarray:
            return self.scores

    args = ("soru", ["b1", "a1"], {"b1": "x", "a1": "y"}, {"b1": 11.0, "a1": 12.0})
    with pytest.raises(ValueError, match="hizalı"):
        compare_rerankings(*args, BadReranker(np.array([0.1])))
    with pytest.raises(ValueError, match="sonlu"):
        compare_rerankings(*args, BadReranker(np.array([0.1, np.nan])))


def test_transformer_reranker_rejects_invalid_batch_size_before_loading_model():
    with pytest.raises(ValueError, match="batch_size"):
        TransformerPageReranker(batch_size=0)
