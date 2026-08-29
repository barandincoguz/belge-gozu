import numpy as np

from belge_gozu.index.store import binarize_pack
from belge_gozu.retrieval.core import ExhaustiveBinaryRetriever, TwoStageRetriever, binary_maxsim
from tests.retrieval.test_core import build_fixture


def test_scores_match_per_page_maxsim():
    """Skor = ham MaxSim / n_q / 128 — normalize [-1,1] ölçek (T14).

    128'e bölme int8/float16 skorlarıyla aynı bandı verir; eşik
    (Settings.min_score_threshold) bu banda göre yorumlanır."""
    idx, meta, embs = build_fixture(n_pages=30)
    r = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    q = embs[17]
    scores = r.score_all(q)
    qp = binarize_pack(q)
    for i in (0, 7, 17, 29):
        expected = binary_maxsim(qp, np.asarray(idx.page_tokens(i))) / q.shape[0] / 128
        assert scores[i] == expected
    # ölçek kilidi: normalize skor bandı dışına çıkmamalı
    assert (scores >= -1.0).all() and (scores <= 1.0).all()


def test_self_match_is_top1():
    idx, meta, embs = build_fixture(n_pages=30)
    r = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    hits = r.search_embedding(embs[17], k=5)
    assert hits[0][0] == 17


def test_exhaustive_recovers_what_stage1_loses():
    """Karışık sorgu (iki sayfanın token'ları): mean-sign Stage-1'in top-1 adayı
    exhaustive argmax'tan sapan bir çift bul ve sapmayı assert et."""
    idx, meta, embs = build_fixture(n_pages=30)
    ex = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    ts = TwoStageRetriever(idx, meta, encoder=None)
    for a in range(30):
        for b in range(a + 1, 30):
            q = np.vstack([embs[a][:4], embs[b][:4]])
            ex_best_i, ex_best_s = ex.search_embedding(q, k=1)[0]
            ts_best_i, ts_best_raw = ts.search_embedding(q, k=1, candidates=1)[0]
            if ts_best_i != ex_best_i:
                # search_embedding RAW döner; aynı ölçeğe getirmek için
                # n_q * 128 (bkz. TwoStageRetriever.search).
                assert ex_best_s >= ts_best_raw / q.shape[0] / 128
                return
    raise AssertionError("hiçbir karışık sorguda stage-1 sapması bulunamadı — fikstürü genişlet")


def test_chunk_boundaries_do_not_change_scores():
    idx, meta, embs = build_fixture(n_pages=30)
    r1 = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    r2 = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    r2.CHUNK_TOKENS = 16  # sayfa başına 8 token -> her chunk ~2 sayfa
    np.testing.assert_array_equal(r1.score_all(embs[3]), r2.score_all(embs[3]))
