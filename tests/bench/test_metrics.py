import pytest

from belge_gozu.bench.metrics import bootstrap_ci, mrr, ndcg_at_k, recall_at_k


def test_recall():
    assert recall_at_k({"a", "b"}, ["a", "x", "y"], 3) == 0.5
    assert recall_at_k({"a"}, [], 5) == 0.0


def test_mrr():
    assert mrr({"a"}, ["x", "a", "y"]) == 0.5
    assert mrr({"a"}, ["x", "y"]) == 0.0


def test_ndcg():
    assert ndcg_at_k({"a"}, ["a", "b"], 5) == pytest.approx(1.0)
    assert ndcg_at_k({"a"}, ["b", "a"], 5) == pytest.approx(0.6309, abs=1e-3)
    assert ndcg_at_k({"a"}, ["b", "c"], 5) == 0.0


def test_bootstrap_ci_deterministic_and_ordered():
    vals = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    lo, hi = bootstrap_ci(vals, n_boot=500, seed=7)
    assert (lo, hi) == bootstrap_ci(vals, n_boot=500, seed=7)
    assert 0.0 <= lo <= sum(vals) / len(vals) <= hi <= 1.0
    assert bootstrap_ci([], n_boot=10) == (0.0, 0.0)
