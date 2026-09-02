"""Geç-etkileşim kanalı ve birleşim getiricisi — saf mantık (469 MB indeks YOK).

CERRAHİ KISIT. Bu değişiklik hiçbir mevcut dosyanın davranışını değiştirmez:
BM25 SAYFA üzerinde kalır (donmuş reçete, `recipe_fingerprint` aynı,
`min_score_threshold` ölçeği aynı). Ölçüm bunun bedava olduğunu gösterdi —
BM25-sayfa + ColBERT birleşimi, BM25-chunk sürümüyle aynı R@50 (0,9362) ve
paraphrase (0,8571) veriyor, R@5'te hatta daha iyi (0,7766 vs 0,7660).

KAPI TEHLİKESİ. `answer/base.py` çekimserliği `hits[0].score < min_score` ile
karar veriyor ve o eşik BM25 ölçeğinde kalibre edildi. ColBERT'in bulduğu bir
sayfa top-1'e girerse taşıdığı BM25 skoru düşük olur ve kapı YANLIŞ sebeple
kapanır. Bu yüzden kanal bayrak arkasındadır ve kalibre bir ColBERT eşiği
olmadan AÇILAMAZ — sessizce yanlış davranmaktansa gürültüyle reddetsin.
"""

import numpy as np
import pytest

from belge_gozu.retrieval.late import (
    LateChannelNotCalibrated,
    LateInteractionChannel,
    chunk_ranking_to_pages,
    validate_index_shapes,
)


class FakeEncoder:
    """Sorguyu sabit bir vektöre kodlar — model ağırlığı gerektirmez."""

    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    def encode_query_vectors(self, text):
        return self.vec


# iki chunk: birincisi [1,0] yönünde, ikincisi [0,1]
EMBS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
OFFSETS = np.array([0, 1, 2], dtype=np.int64)
CIDS = ["d:m1", "d:m2"]
PAGES = {"d:m1": ("d:1",), "d:m2": ("d:2", "d:3")}


def channel(vec):
    return LateInteractionChannel(
        embeddings=EMBS, offsets=OFFSETS, chunk_ids=CIDS,
        chunk_pages=PAGES, encoder=FakeEncoder(vec),
    )


# --------------------------------------------------------------------------
# sıralama
# --------------------------------------------------------------------------


def test_ranks_the_chunk_whose_vectors_match_the_query():
    assert channel([[1.0, 0.0]]).rank_chunks("s")[0] == "d:m1"
    assert channel([[0.0, 1.0]]).rank_chunks("s")[0] == "d:m2"


def test_rank_chunks_returns_every_chunk():
    assert sorted(channel([[1.0, 0.0]]).rank_chunks("s")) == sorted(CIDS)


def test_candidate_pages_expands_multi_page_chunks_in_order():
    assert channel([[0.0, 1.0]]).candidate_pages("s") == ["d:2", "d:3", "d:1"]


def test_candidate_pages_deduplicates():
    ch = LateInteractionChannel(
        embeddings=EMBS, offsets=OFFSETS, chunk_ids=CIDS,
        chunk_pages={"d:m1": ("d:1",), "d:m2": ("d:1", "d:2")},
        encoder=FakeEncoder([[0.0, 1.0]]),
    )
    assert ch.candidate_pages("s") == ["d:1", "d:2"]


def test_candidate_pages_respects_limit():
    assert len(channel([[0.0, 1.0]]).candidate_pages("s", limit=1)) == 2  # tek chunk, iki sayfa


# --------------------------------------------------------------------------
# chunk -> sayfa indirgeme (saf yardımcı)
# --------------------------------------------------------------------------


def test_chunk_ranking_to_pages_keeps_first_occurrence():
    out = chunk_ranking_to_pages(["a", "b"], {"a": ("p1", "p2"), "b": ("p2", "p3")})
    assert out == ["p1", "p2", "p3"]


def test_chunk_ranking_to_pages_ignores_unknown_chunk():
    assert chunk_ranking_to_pages(["a", "zzz"], {"a": ("p1",)}) == ["p1"]


# --------------------------------------------------------------------------
# artefakt tutarlılığı — sessiz hizasızlık en tehlikeli hata
# --------------------------------------------------------------------------


def test_validate_index_shapes_accepts_consistent_artifact():
    validate_index_shapes(EMBS, OFFSETS, CIDS)


def test_validate_index_shapes_rejects_chunk_id_count_mismatch():
    with pytest.raises(ValueError, match="chunk_ids"):
        validate_index_shapes(EMBS, OFFSETS, ["only-one"])


def test_validate_index_shapes_rejects_offset_tail_mismatch():
    """Son ofset vektör sayısına eşit değilse indeks kırık demektir."""
    with pytest.raises(ValueError, match="ofset"):
        validate_index_shapes(EMBS, np.array([0, 1, 99], dtype=np.int64), CIDS)


def test_validate_index_shapes_rejects_duplicate_chunk_ids():
    """`rg1935a:m1` 21 kez tekrarlanıyordu ve 22 sayfayı erişilemez yapmıştı."""
    with pytest.raises(ValueError, match="benzersiz"):
        validate_index_shapes(EMBS, OFFSETS, ["d:m1", "d:m1"])


# --------------------------------------------------------------------------
# kalibrasyon kapısı — kanal kalibre eşik olmadan AÇILAMAZ
# --------------------------------------------------------------------------


def test_enabling_the_channel_without_a_calibrated_threshold_is_refused():
    """Çekimserlik eşiği BM25 ölçeğinde kalibre; ColBERT'in kendi eşiği yok.

    Kanal açıkken ColBERT'in bulduğu bir sayfa top-1'e girerse taşıdığı BM25
    skoru düşük olur ve kapı YANLIŞ sebeple kapanır. Sessiz yanlış davranış
    yerine gürültülü ret.
    """
    from belge_gozu.retrieval.late import require_calibrated_late_channel

    with pytest.raises(LateChannelNotCalibrated, match="kalibre"):
        require_calibrated_late_channel(enabled=True, calibrated_threshold=None)


def test_disabled_channel_needs_no_calibration():
    from belge_gozu.retrieval.late import require_calibrated_late_channel

    require_calibrated_late_channel(enabled=False, calibrated_threshold=None)


def test_enabled_channel_with_a_calibrated_threshold_is_allowed():
    from belge_gozu.retrieval.late import require_calibrated_late_channel

    require_calibrated_late_channel(enabled=True, calibrated_threshold=12.5)
