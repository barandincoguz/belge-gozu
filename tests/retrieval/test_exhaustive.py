import numpy as np

from belge_gozu.index.store import binarize_pack
from belge_gozu.retrieval.core import ExhaustiveBinaryRetriever, TwoStageRetriever, binary_maxsim
from tests.retrieval.test_core import build_fixture


def test_scores_match_per_page_maxsim():
    idx, meta, embs = build_fixture(n_pages=30)
    r = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    q = embs[17]
    scores = r.score_all(q)
    qp = binarize_pack(q)
    for i in (0, 7, 17, 29):
        expected = binary_maxsim(qp, np.asarray(idx.page_tokens(i))) / q.shape[0]
        assert scores[i] == expected


def test_self_match_is_top1():
    idx, meta, embs = build_fixture(n_pages=30)
    r = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    hits = r.search_embedding(embs[17], k=5)
    assert hits[0][0] == 17


def test_exhaustive_beats_broken_stage1_counterexample():
    """Stage-1'in kaybettiği sonucu exhaustive bulur: Stage-1'i top-1 aday ile
    kısıtla; exhaustive tüm sayfaları görür."""
    idx, meta, embs = build_fixture(n_pages=30)
    ex = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    ts = TwoStageRetriever(idx, meta, encoder=None)
    q = embs[23]
    ex_top = ex.search_embedding(q, k=30)
    ts_top = ts.search_embedding(q, k=30, candidates=1)
    assert len(ex_top) == 30 and len(ts_top) == 1
    assert ex_top[0][0] == 23


def test_chunk_boundaries_do_not_change_scores():
    idx, meta, embs = build_fixture(n_pages=30)
    r1 = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    r2 = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    r2.CHUNK_TOKENS = 16  # sayfa başına 8 token -> her chunk ~2 sayfa
    np.testing.assert_array_equal(r1.score_all(embs[3]), r2.score_all(embs[3]))
