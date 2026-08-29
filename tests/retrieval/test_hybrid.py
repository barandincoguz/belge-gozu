"""HybridRetriever: sıralamayı metin kanalı belirler, görsel kanal telemetride."""

import numpy as np
import pandas as pd
import pytest

from belge_gozu.answer.base import ABSTAIN_TEXT, Answer, AskService
from belge_gozu.index.store import PackedIndex
from belge_gozu.retrieval.hybrid import HybridRetriever
from belge_gozu.retrieval.text import BM25Index, extract_doc_name_tokens
from belge_gozu.telemetry.collect import collecting

IDS = ["k1:1", "k1:2", "k1:3", "k2:1", "k2:2", "k3:1"]
TEXTS = [
    "TÜRK MEDENİ KANUNU\nKanun Numarası: 4721\n",
    "MADDE 19 - Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir.\n",
    "MADDE 24 - Kişilik hakkı saldırıya uğrayan kimse hâkimden koruma isteyebilir.\n",
    "İŞ KANUNU\nKanun Numarası: 4857\n",
    "MADDE 53 - Yıllık ücretli izin süresi hizmet süresine göre belirlenir.\n",
    "TÜRK CEZA KANUNU\nSuçta ve cezada kanunilik ilkesi.\n",
]
Q_ROUTED = "Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?"


class _FixedEncoder:
    """encode_query her zaman AYNI embedding'i döner (görsel kanal deterministik)."""

    def __init__(self, emb: np.ndarray):
        self.emb = emb

    def encode_pages(self, images):
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.emb


def _fixture(encoder=None):
    rng = np.random.default_rng(5)
    embs = [rng.standard_normal((8, 128)).astype(np.float32) for _ in IDS]
    index = PackedIndex.build(IDS, embs)
    meta = pd.DataFrame(
        {
            "page_id": IDS,
            "doc_id": [pid.partition(":")[0] for pid in IDS],
            "doc_name": [f"Belge {pid}" for pid in IDS],
            "doc_type": ["kanun"] * len(IDS),
            "source_url": ["https://example.org"] * len(IDS),
            "page_no": [int(pid.partition(":")[2]) for pid in IDS],
            "image_path": [f"images/{pid.replace(':', '/')}.webp" for pid in IDS],
        }
    )
    bm25 = BM25Index(IDS, TEXTS)
    names = extract_doc_name_tokens(IDS, TEXTS)
    enc = encoder if encoder is not None else _FixedEncoder(embs[4])
    return HybridRetriever(index, meta, enc, bm25, names), bm25, embs


def _bm25_order(bm25: BM25Index, q: str) -> list[str]:
    return [IDS[i] for i in np.argsort(-bm25.scores(q), kind="stable")]


def test_ranking_follows_bm25_and_routing():
    """Reçete sırası: BM25 azalan + adı sorguda geçen dokümanın sayfaları öne."""
    r, bm25, _ = _fixture()
    assert r.routed_docs(Q_ROUTED) == {"k1"}  # ad token'ı {"meden"} sorguda geçiyor
    plain = _bm25_order(bm25, Q_ROUTED)
    ids = [h.page_id for h in r.search(Q_ROUTED, k=len(IDS))]

    assert set(ids) == set(plain)  # pencere kümesi değişmez
    # yönlendirme GERÇEKTEN iz bırakıyor: k1:3 saf BM25'te geride, hibritte önde
    assert plain != ids
    assert ids.index("k1:3") < plain.index("k1:3")
    # yönlendirilen dokümanın TÜM sayfaları önde, kendi BM25 sıralarını koruyarak
    routed_pages = [pid for pid in plain if pid.startswith("k1:")]
    assert ids[: len(routed_pages)] == routed_pages
    assert ids[len(routed_pages) :] == [pid for pid in plain if not pid.startswith("k1:")]


def test_page_hit_score_is_bm25_scale():
    """PageHit.score = O SAYFANIN BM25 skoru (görsel normalize [-1,1] değil)."""
    r, bm25, _ = _fixture()
    q = "yerleşim yeri nedir"
    scores = dict(zip(IDS, bm25.scores(q).tolist(), strict=True))
    hits = r.search(q, k=5)
    for h in hits:
        assert h.score == pytest.approx(scores[h.page_id])
    assert hits[0].score > 1.5  # BM25 bandı; normalize [-1,1] eşiğinin üstünde
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_visual_channel_does_not_change_ranking():
    """Görsel kanal koşar ama sıraya GİRMEZ: encoder değişse de sıra aynı."""
    r_a, _, embs = _fixture()
    r_b, _, _ = _fixture(encoder=_FixedEncoder(embs[0]))
    q = "yıllık ücretli izin süresi"
    assert [h.page_id for h in r_a.search(q, k=5)] == [h.page_id for h in r_b.search(q, k=5)]


def test_zero_match_query_scores_zero_and_abstains_at_service_level():
    """Korpusta hiç geçmeyen sorgu -> top skor 0 -> eşiğin altında -> abstain."""

    class _Answerer:
        def answer(self, question, pages, image_loader):
            raise AssertionError("eşik altı sorguda yanıtlayıcı çağrılmamalı")

    r, _, _ = _fixture()
    hits = r.search("asdf qwerty zxcvbn", k=5)
    assert hits and hits[0].score == 0.0
    svc = AskService(r, _Answerer(), min_score=10.6, image_loader=lambda p: b"x")
    answer, hits = svc.ask("asdf qwerty zxcvbn", k=5)
    assert answer.abstained and answer.text == ABSTAIN_TEXT and answer.citations == []


def test_search_records_all_four_stages():
    """Aşama adları detail.stages'e düşer (telemetri fallback'i bunları taşır)."""
    r, _, _ = _fixture()
    with collecting() as col:
        r.search("yerleşim yeri", k=3)
    assert {"query_encode", "exhaustive_maxsim", "text_bm25", "route_fuse"} <= set(col.stages)


def test_last_retrieval_meta_carries_both_channel_tops_and_routing():
    r, bm25, _ = _fixture()
    r.search(Q_ROUTED, k=5)
    meta = r.last_retrieval_meta
    assert meta is not None
    assert meta["routed_docs"] == ["k1"]
    # `bm25_top1` KANALIN tepesi (yönlendirme sonrası servis edilen top-1 ayrı
    # bir sayfa olabilir; o olayın `top_score` alanında durur).
    assert meta["bm25_top1"] == pytest.approx(float(bm25.scores(Q_ROUTED).max()))
    assert isinstance(meta["visual_top1"], float)


def test_last_retrieval_meta_is_request_scoped():
    """Künye ContextVar'da: her bağlam kendi değerini görür, istekler karışmaz."""
    import contextvars

    r, _, _ = _fixture()
    r.search(Q_ROUTED, k=1)
    assert r.last_retrieval_meta["routed_docs"] == ["k1"]

    def _other():
        r.search("asdf qwerty", k=1)
        return r.last_retrieval_meta["routed_docs"]

    assert contextvars.copy_context().run(_other) == []
    # dış bağlamdaki künye BOZULMADI (iç bağlamın yazması dışarı sızmaz)
    assert r.last_retrieval_meta["routed_docs"] == ["k1"]


def test_rank_all_matches_search_order_without_encoder():
    """rank_all model gerektirmez (görsel kanalı koşmaz) ama AYNI sırayı verir."""
    r, _, _ = _fixture()
    q = "İş Kanunu yıllık ücretli izin"
    assert r.rank_all(q)[:5] == [h.page_id for h in r.search(q, k=5)]

    r_noenc, bm25, _ = _fixture(encoder=None)
    r_noenc.encoder = None
    assert r_noenc.rank_all(q) == r.rank_all(q)
    with pytest.raises(RuntimeError, match="encoder"):
        r_noenc.search(q, k=1)


def test_misaligned_text_index_is_rejected():
    """Metin ve görsel indeksin page_ids'i birebir eşleşmeli — sessiz kayma yok."""
    r, _, _ = _fixture()
    shuffled = BM25Index(list(reversed(IDS)), list(reversed(TEXTS)))
    with pytest.raises(ValueError, match="page_ids"):
        HybridRetriever(r.index, r.meta.reset_index(drop=True), r.encoder, shuffled, r.doc_names)


def test_k_limits_returned_hits():
    r, _, _ = _fixture()
    assert len(r.search("kanun", k=2)) == 2
    assert len(r.search("kanun", k=99)) == len(IDS)


def test_answer_path_uses_hits_when_above_threshold():
    """Eşik üstü: yanıtlayıcı BM25'in seçtiği sayfalarla çağrılır."""

    class _Answerer:
        def answer(self, question, pages, image_loader):
            return Answer(text="ok", citations=[pages[0].page_id])

    r, _, _ = _fixture()
    svc = AskService(r, _Answerer(), min_score=0.5, image_loader=lambda p: b"x")
    answer, hits = svc.ask("yerleşim yeri nedir", k=3)
    assert not answer.abstained and answer.citations == [hits[0].page_id]
