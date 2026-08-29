"""Metin kanalı birim testleri — reçete sabitleri ve sözleşmeleri.

Reçete ÖLÇÜLMÜŞ bir bütündür (findings 2026-08-29-autoresearch-text-channel.md):
tek tek parçaları "iyileştirmek" ölçümü geçersiz kılar. Bu yüzden burada
davranış değil, SABİTLER ve YAPISAL SÖZLEŞMELER kilitlenir.
"""

import math

import numpy as np
import pytest

from belge_gozu.retrieval.text import (
    F5,
    STOPWORDS,
    WINDOW,
    BM25Index,
    extract_doc_name_tokens,
    route_window,
    tokenize,
    tr_lower,
)


def test_recipe_constants_are_the_measured_ones():
    """Sabitler exp7 reçetesinden: F5=5, pencere=20 (R@20 guardrail'iyle hizalı)."""
    assert F5 == 5
    assert WINDOW == 20
    # İçerik taşıyabilecek kelimeler listede OLMAMALI (canary'ye ayar riski).
    assert {"ve", "göre", "nedir", "sayılı"} <= STOPWORDS
    assert not ({"zaman", "iş", "izin", "yerleşim"} & STOPWORDS)


def test_tr_lower_handles_dotted_and_dotless_i():
    """Python'un lower()'ı 'I'yı 'i' yapar; Türkçede 'ı' olmalı."""
    assert tr_lower("İSTANBUL") == "istanbul"
    assert tr_lower("IRAK") == "ırak"
    assert "IRAK".lower() != tr_lower("IRAK")  # farkın gerçek olduğunu göster


def test_tokenize_known_cases():
    # stopword eleme TAM KELİME üzerinde ve KIRPMADAN ÖNCE: "göre" düşer,
    # "görev" (F5 -> "görev") kalır — kırpma sonrası ikisi ayrışamazdı.
    assert tokenize("İş Kanunu'na göre") == ["iş", "kanun", "na"]
    assert tokenize("görev göre") == ["görev"]
    # F5 ön-ek kırpması
    assert tokenize("yerleşim yerleşimi") == ["yerle", "yerle"]
    # tek harfli parçalar elenir (>=2 karakter kuralı)
    assert tokenize("a bc d") == ["bc"]
    # tamamı işlev kelimesi olan sorgu boş token listesi verir
    assert tokenize("ve veya ile için") == []
    # noktalama \w+ ile ayrılır
    assert tokenize("Kanunu'na") == ["kanun", "na"]


# --- BM25: elle hesaplanmış küçük korpus -----------------------------------
#
# d0: [kira, kira, borç]  d1: [kira, ceza]  d2: [vergi]
# n=3, avgdl=2.0, df(kira)=2, df(borç)=df(ceza)=df(vergi)=1
# idf(t) = log(1 + (n-df+0.5)/(df+0.5))
_IDS = ["d0:1", "d1:1", "d2:1"]
_TEXTS = ["kira kira borç", "kira ceza", "vergi"]
_IDF_KIRA = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
_IDF_TEK = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))


def test_bm25_scores_match_hand_computation():
    idx = BM25Index(_IDS, _TEXTS)
    assert idx.avgdl == pytest.approx(2.0)
    assert idx.idf["kira"] == pytest.approx(_IDF_KIRA)
    assert idx.idf["borç"] == pytest.approx(_IDF_TEK)

    k1, b = 1.5, 0.75
    # d0: f=2, dl=3 ; d1: f=1, dl=2 ; d2: yok
    d0 = _IDF_KIRA * 2 * (k1 + 1) / (2 + k1 * (1 - b + b * 3 / 2.0))
    d1 = _IDF_KIRA * 1 * (k1 + 1) / (1 + k1 * (1 - b + b * 2 / 2.0))
    got = idx.scores("kira")
    assert got.shape == (3,)
    assert got[0] == pytest.approx(d0, rel=1e-5)
    assert got[1] == pytest.approx(d1, rel=1e-5)
    assert got[2] == 0.0
    # elle hesaplanan sayılar (regresyon kilidi)
    assert float(got[0]) == pytest.approx(0.578466, rel=1e-4)
    assert float(got[1]) == pytest.approx(0.470004, rel=1e-4)


def test_bm25_scores_are_aligned_to_page_ids():
    """Skor vektörü page_ids SIRASINDA — bir kayma yanlış sayfayı döndürür."""
    idx = BM25Index(_IDS, _TEXTS)
    s = idx.scores("vergi")
    assert idx.page_ids == _IDS
    assert s.argmax() == 2 and s[0] == 0.0 and s[1] == 0.0


def test_bm25_unknown_query_token_scores_zero():
    """Korpusta hiç geçmeyen sorgu -> tümü 0 (üst katman eşikte abstain'e düşer)."""
    idx = BM25Index(_IDS, _TEXTS)
    assert not idx.scores("qwerty zxcvbn").any()


def test_bm25_rejects_misaligned_or_empty_input():
    with pytest.raises(ValueError, match="eşleşmiyor"):
        BM25Index(["a:1", "b:1"], ["tek metin"])
    with pytest.raises(ValueError, match="boş korpus"):
        BM25Index([], [])


# --- doküman adı ------------------------------------------------------------


def test_doc_name_tokens_from_page_one_title_line():
    """Ad 1. sayfanın büyük-harfli başlık satırından; jenerik parçalar atılır."""
    names = extract_doc_name_tokens(
        ["k4721:1", "k4721:4", "k4857:1"],
        [
            "T.C.\nTÜRK MEDENİ KANUNU\nKanun Numarası: 4721\n",
            "MADDE 19 - Yerleşim yeri bir kimsenin...\n",
            "İŞ KANUNU\nKanun Numarası: 4857\n",
        ],
    )
    # "türk"/"kanun" jenerik listesinde -> düşer
    assert names == {"k4721": frozenset({"meden"}), "k4857": frozenset({"iş"})}


def test_doc_name_ignores_non_first_pages_and_untitled_docs():
    names = extract_doc_name_tokens(
        ["a:2", "b:1", "c:1"],
        [
            "TÜRK TİCARET KANUNU\n",  # 1. sayfa değil -> yok sayılır
            "başlık satırı küçük harf, kanun geçse de eşleşmez\n",
            "TÜRK KANUNU\n",  # ad token'ları tamamen jenerik -> eklenmez
        ],
    )
    assert names == {}


# --- pencere-içi yönlendirme ------------------------------------------------


def _ranking(n: int = 30) -> list[str]:
    # d0:1..d0:10 aralıklı serpiştirilmiş; ilk 20 pencere, kalanı dokunulmaz
    return [f"d{i % 3}:{i}" for i in range(n)]


def test_route_window_keeps_top_window_set_and_tail_identical():
    """Yapısal sözleşme: pencere KÜMESİ ve pencere SONRASI hiç değişmez."""
    ranking = _ranking()
    out = route_window(ranking, {"d1"}, window=20)
    assert len(out) == len(ranking)
    assert set(out[:20]) == set(ranking[:20])  # R@20 tanım gereği korunur
    assert out[20:] == ranking[20:]  # pencere sonrası dokunulmaz


def test_route_window_puts_routed_first_preserving_order():
    ranking = ["a:1", "b:1", "a:2", "c:1", "b:2"]
    out = route_window(ranking, {"b"}, window=5)
    assert out == ["b:1", "b:2", "a:1", "a:2", "c:1"]


def test_route_window_without_routed_docs_is_identity():
    ranking = _ranking()
    assert route_window(ranking, set(), window=20) == ranking


def test_route_window_does_not_pull_pages_into_the_window():
    """Pencere DIŞINDAKİ yönlendirilmiş sayfa öne ÇEKİLMEZ (exp6'nın vetolanan
    davranışı): aday kümesini değiştirmek R@20'yi düşürüyordu."""
    ranking = [f"x:{i}" for i in range(20)] + ["hedef:1"]
    out = route_window(ranking, {"hedef"}, window=20)
    assert out == ranking
    assert out[20] == "hedef:1"


def test_route_window_default_is_the_measured_window():
    ranking = _ranking()
    assert route_window(ranking, {"d1"}) == route_window(ranking, {"d1"}, window=WINDOW)


def test_bm25_ranking_plus_routing_matches_recipe_order():
    """Uçtan uca küçük örnek: BM25 sırası + pencere-içi yönlendirme."""
    ids = ["k1:1", "k1:9", "k2:1"]
    texts = ["ÖRNEK KANUNU\nkira", "kira kira kira sözleşme", "ÖTEKİ KANUNU\nkira"]
    idx = BM25Index(ids, texts)
    scores = idx.scores("Örnek Kanunu kira")
    order = [ids[i] for i in np.argsort(-scores, kind="stable")]
    names = extract_doc_name_tokens(ids, texts)
    routed = {d for d, t in names.items() if t <= set(tokenize("Örnek Kanunu kira"))}
    assert routed == {"k1"}
    out = route_window(order, routed, window=3)
    assert out[:2] == [pid for pid in order if pid.startswith("k1")]
    assert set(out) == set(order)
