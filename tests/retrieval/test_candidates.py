import pytest

from belge_gozu.retrieval.candidates import build_candidate_pool


def test_pool_uses_each_source_at_its_own_depth_and_deduplicates():
    assert build_candidate_pool(
        ["b1", "b2", "b3"],
        [["a1", "b2", "a2"], ["c1", "a1", "c2"]],
        limit=2,
    ) == ["b1", "b2", "a1", "c1"]


def test_pool_keeps_bm25_first_even_when_late_repeats_it():
    assert build_candidate_pool(["b1", "b2"], [["b1", "a1"]], limit=50)[0] == "b1"


def test_pool_rejects_non_positive_depth():
    with pytest.raises(ValueError, match="limit"):
        build_candidate_pool(["b1"], [["a1"]], limit=0)
