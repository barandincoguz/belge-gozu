"""Metin kanalı birim testleri — reçete sabitleri ve sözleşmeleri.

Reçete ÖLÇÜLMÜŞ bir bütündür (findings 2026-08-29-autoresearch-text-channel.md):
tek tek parçaları "iyileştirmek" ölçümü geçersiz kılar. Bu yüzden burada
davranış değil, SABİTLER ve YAPISAL SÖZLEŞMELER kilitlenir.
"""

import math
import random
import re

import numpy as np
import pytest

from belge_gozu.retrieval.text import (
    F5,
    QTF_CAP,
    STOPWORDS,
    WINDOW,
    BM25Index,
    ascii_fold,
    extract_doc_name_tokens,
    rank_order,
    recipe_fingerprint,
    route_window,
    routed_docs,
    tokenize,
    tr_lower,
)


def test_recipe_constants_are_the_measured_ones():
    """Sabitler reçeteden: F5=5 (exp3), pencere=50 (exp8 — journal #8).

    Pencere exp7'de 20'ydi (R@20 guardrail'iyle yapısal hizalı); exp8 ölçümü
    50'ye çıkardı ve R@20 korunmakla kalmayıp 0.907 -> 0.9302'ye YÜKSELDİ,
    R@5 0.8140 -> 0.8372 (+c214)."""
    assert F5 == 5
    assert WINDOW == 50
    # İçerik taşıyabilecek kelimeler listede OLMAMALI (retrieval_eval'ye ayar riski).
    assert {"ve", "göre", "nedir", "sayılı"} <= STOPWORDS
    assert not ({"zaman", "iş", "izin", "yerleşim"} & STOPWORDS)


def test_ascii_fold_maps_the_measured_character_set():
    """Katlama tablosu reçeteden (exp12): çğıöşü + düzeltme işaretli âîû.

    Katlama UZUNLUĞU değiştirmez — F5 kırpma sınırı iki yazımda da aynı yere
    düşsün diye (tokenize'ın sıra sözleşmesi buna dayanır)."""
    assert ascii_fold("çğıöşüâîû") == "cgiosuaiu"
    assert ascii_fold("hâkim") == "hakim"
    assert len(ascii_fold("yıllık ücretli izin")) == len("yıllık ücretli izin")
    # ASCII girdi değişmez (aksansız yazan kullanıcı katlamadan etkilenmez)
    assert ascii_fold("yillik ucretli izin") == "yillik ucretli izin"


def test_tr_lower_handles_dotted_and_dotless_i():
    """Python'un lower()'ı 'I'yı 'i' yapar; Türkçede 'ı' olmalı."""
    assert tr_lower("İSTANBUL") == "istanbul"
    assert tr_lower("IRAK") == "ırak"
    assert "IRAK".lower() != tr_lower("IRAK")  # farkın gerçek olduğunu göster


def test_tokenize_known_cases():
    # exp12: çıktı KATLANMIŞ uzayda ("iş" -> "is", "görev" -> "gorev").
    # stopword eleme TAM KELİME üzerinde ve KIRPMADAN ÖNCE: "göre" düşer,
    # "görev" (F5 -> "gorev") kalır — kırpma sonrası ikisi ayrışamazdı.
    assert tokenize("İş Kanunu'na göre") == ["is", "kanun", "na"]
    assert tokenize("görev göre") == ["gorev"]
    # F5 ön-ek kırpması
    assert tokenize("yerleşim yerleşimi") == ["yerle", "yerle"]
    # tek harfli parçalar elenir (>=2 karakter kuralı)
    assert tokenize("a bc d") == ["bc"]
    # tamamı işlev kelimesi olan sorgu boş token listesi verir
    assert tokenize("ve veya ile için") == []
    # noktalama \w+ ile ayrılır
    assert tokenize("Kanunu'na") == ["kanun", "na"]


def test_tokenize_is_writing_invariant():
    """exp12'nin ÜRÜN ÖZELLİĞİ: aksanlı ve aksansız yazım AYNI token'ları verir.

    Ölçüm (journal #11-#12): katlama olmadan aksansız sorgularda R@5
    0.8372 -> 0.5814 çöküyordu; iki taraflı katlamayla iki koşulda da
    0.8605 (37/43). Bu test o davranışın birim düzeyindeki kilididir."""
    accented = tokenize("İş Kanunu'na göre yıllık ücretli izin süresi ne kadardır?")
    plain = tokenize("Is Kanununa gore yillik ucretli izin suresi ne kadardir?")
    assert tokenize("yillik ucretli izin") == tokenize("yıllık ücretli izin")
    # aksansız yazımın ÜRETTİĞİ her token aksanlı yazımda da var (tam kesişim)
    assert set(plain) <= set(accented)
    assert set(plain) == {"is", "kanun", "yilli", "ucret", "izin", "sures", "kadar"}


def test_tokenize_folds_function_words_too():
    """Stopword eleme KATLANMIŞ uzayda: "göre"/"gore", "için"/"icin" ikisi de düşer.

    Katlanmamış bir stoplist yazım-değişmezliği yarım bırakırdı — aksansız
    yazan kullanıcının işlev kelimeleri içerik token'ı sayılıp IDF'i
    kirletirdi."""
    assert tokenize("ne icin nasil kac") == []
    assert tokenize("ne için nasıl kaç") == []


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
    # exp12: sözlük anahtarları KATLANMIŞ uzayda ("borç" -> "borc"). Skorların
    # SAYISAL değerleri değişmez — katlama df/dl/idf hesabına değil yalnız
    # token kimliğine dokunur.
    assert "borç" not in idx.idf
    assert idx.idf["borc"] == pytest.approx(_IDF_TEK)

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


def test_repeated_query_term_cannot_inflate_the_score():
    """Y1 SALDIRI TESTİ: aynı terimi 80 kez yazmak skoru en fazla ~2× yapar.

    Canlı sondaj (2026-08-30): `"ihbar "×80` üretimde top-1'i 667.50'ye
    çıkarıyordu — eşiğin (10.6) 63 katı. `MAX_QUERY_CHARS=500` bunu
    KAPATMIYOR, yalnız ölçekliyor. Sınır 2.05× (tam tavan 2.00×; pay float32
    yuvarlamasına bırakıldı).
    """
    ids = ["k1:1", "k1:2", "k2:1"]
    texts = [
        "ihbar süresi ve ihbar tazminatı hakkında hüküm",
        "yıllık ücretli izin süresi",
        "ihbar önelleri tablosu",
    ]
    idx = BM25Index(ids, texts)
    once = idx.scores("ihbar")
    flood = idx.scores(" ".join(["ihbar"] * 80))
    assert once.max() > 0  # test gerçekten bir şey ölçüyor
    assert flood.max() <= 2.05 * once.max()
    # Ve tavan tam olarak QTF_CAP: 80 tekrar 2 tekrarla AYNI skoru verir.
    assert np.allclose(flood, idx.scores(" ".join(["ihbar"] * QTF_CAP)))


def test_query_term_saturation_cap_is_the_measured_one():
    """Tavan 2 (exp14). 1 olsaydı meşru vurgu ("artış ... artış") cezalanırdı;
    tavansız hâli Y1'in ta kendisidir."""
    assert QTF_CAP == 2
    idx = BM25Index(["a:1"], ["kira artışı oranı"])
    s1 = float(idx.scores("kira")[0])
    s2 = float(idx.scores("kira kira")[0])
    s3 = float(idx.scores("kira kira kira")[0])
    assert s2 == pytest.approx(2 * s1, rel=1e-5)
    assert s3 == pytest.approx(s2, rel=1e-6)


def test_unique_query_terms_are_unaffected_by_the_cap():
    """Reçete PARİTESİ: hiçbir terimi tekrar etmeyen sorgu (retrieval_eval'nin
    tamamı) tavan öncesi/sonrası birebir aynı skoru alır — exp14'ün
    R@5 0.8605 / MRR 0.632 pariteyi ölçen koşumunun birim karşılığı."""
    ids = ["a:1", "b:1"]
    texts = ["yıllık ücretli izin süresi", "ihbar tazminatı"]
    idx = BM25Index(ids, texts)
    q = "yıllık ücretli izin süresi nedir"
    toks = tokenize(q)
    assert len(set(toks)) == len(toks)  # önkoşul: tekrar YOK
    manual = np.zeros(len(ids), dtype=np.float32)
    for tok in toks:
        idf = idx.idf.get(tok)
        if idf is None:
            continue
        for i, freqs in enumerate(idx.doc_freqs):
            f = freqs.get(tok)
            if f:
                dl = idx.doc_lens[i]
                manual[i] += (
                    idf * f * (idx.k1 + 1) / (f + idx.k1 * (1 - idx.b + idx.b * dl / idx.avgdl))
                )
    assert np.allclose(idx.scores(q), manual)


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
    """Ad 1. sayfanın büyük-harfli başlık satırından; jenerik parçalar atılır.

    Ad çıkarımı `tokenize`'ı kullandığı için exp12'den sonra KATLANMIŞ uzayda
    çalışır ("iş" -> "is") — yani sorgu tarafıyla aynı uzayda, yönlendirme
    aksansız sorguda da tetiklenir.

    "turk" ÇIKMAZ: `_GENERIC` listesi AKSANLI ("türk") tutulduğu için katlanmış
    "turk" token'ı elenmez. Bu, ölçülen reçetenin (0.8605) davranışıdır ve
    bilinçle korunmuştur — bkz. `retrieval/text.py` `_GENERIC` yorumu."""
    names = extract_doc_name_tokens(
        ["k4721:1", "k4721:4", "k4857:1"],
        [
            "T.C.\nTÜRK MEDENİ KANUNU\nKanun Numarası: 4721\n",
            "MADDE 19 - Yerleşim yeri bir kimsenin...\n",
            "İŞ KANUNU\nKanun Numarası: 4857\n",
        ],
    )
    assert names == {"k4721": frozenset({"turk", "meden"}), "k4857": frozenset({"is"})}


def test_doc_name_ignores_non_first_pages_and_untitled_docs():
    names = extract_doc_name_tokens(
        ["a:2", "b:1", "c:1"],
        [
            "TÜRK TİCARET KANUNU\n",  # 1. sayfa değil -> yok sayılır
            "başlık satırı küçük harf, kanun geçse de eşleşmez\n",
            # ad token'ları ("cumhu", "kanun") TAMAMEN jenerik -> eklenmez.
            # (Katlama sonrası "türk" artık jenerik listesiyle eşleşmediği için
            # örnek "TÜRK KANUNU" değil "CUMHURİYET KANUNU" ile kuruldu.)
            "CUMHURİYET KANUNU\n",
        ],
    )
    assert names == {}


# --- pencere-içi yönlendirme ------------------------------------------------


def _ranking(n: int = 30) -> list[str]:
    # d0:1..d0:10 aralıklı serpiştirilmiş; ilk 20 pencere, kalanı dokunulmaz
    return [f"d{i % 3}:{i}" for i in range(n)]


def test_route_window_keeps_top_window_set_and_tail_identical():
    """Yapısal sözleşme (property): pencere KÜMESİ ve pencere SONRASI hiç değişmez.

    Tek bir örnekle değil, rastgele üretilmiş 300 (sıralama, yönlendirilen
    küme, pencere) üçlüsüyle sınanır — sözleşme `window` değerinden BAĞIMSIZ
    olarak yapısaldır ve exp8'in 20 -> 50 değişikliğinden sonra da aynen
    geçerlidir. VARSAYILAN pencere de taranan değerler arasında."""
    rng = random.Random(20260830)
    docs = [f"d{i}" for i in range(6)]
    for _ in range(300):
        n = rng.randint(1, 120)
        ranking = [f"{rng.choice(docs)}:{i}" for i in range(n)]
        routed = set(rng.sample(docs, rng.randint(0, len(docs))))
        window = rng.choice([0, 1, 2, 5, 20, WINDOW, n, n + 10])

        out = route_window(ranking, routed, window=window)

        assert len(out) == len(ranking)
        assert set(out) == set(ranking)  # hiçbir sayfa kaybolmuyor/eklenmiyor
        assert set(out[:window]) == set(ranking[:window])  # R@window tanım gereği korunur
        assert out[window:] == ranking[window:]  # pencere sonrası dokunulmaz
        # pencere İÇİNDE: yönlendirilenler önce, iki grupta da BM25 sırası korunuyor
        win = ranking[:window]
        front = [p for p in win if p.partition(":")[0] in routed]
        back = [p for p in win if p.partition(":")[0] not in routed]
        assert out[:window] == (front + back if routed else win)


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
    ranking = [f"x:{i}" for i in range(WINDOW)] + ["hedef:1"]
    out = route_window(ranking, {"hedef"})
    assert out == ranking
    assert out[WINDOW] == "hedef:1"


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


# --- reçete parmak izi + yönlendirme yüklemi (P2 T6 versiyonlama) -----------


def test_routed_docs_predicate_is_the_hybrid_one():
    """Yüklem tek yerde: adının jenerik-dışı TÜM token'ları sorguda geçen doküman."""
    names = {"k1": frozenset({"meden"}), "k2": frozenset({"icra", "iflas"})}
    assert routed_docs("Medeni Kanun madde 19", names) == {"k1"}
    assert routed_docs("icra takibi", names) == set()  # "iflas" eksik -> tetiklenmez
    assert routed_docs("icra ve iflas hukuku", names) == {"k2"}
    assert routed_docs("hiçbir ad geçmiyor", names) == set()


def test_recipe_fingerprint_is_a_stable_short_hex():
    fp = recipe_fingerprint()
    assert len(fp) == 12
    assert all(c in "0123456789abcdef" for c in fp)
    assert fp == recipe_fingerprint()  # deterministik


@pytest.mark.parametrize(
    ("const", "value"),
    [
        ("F5", 6),
        ("WINDOW", 20),
        ("QTF_CAP", 1),
        ("K1", 1.2),
        ("B", 0.5),
        ("MIN_TOKEN_CHARS", 3),
        ("STOPWORDS", frozenset({"ve"})),
        ("_GENERIC", frozenset({"kanun"})),
        ("RECIPE_VERSION", 2),
        # review m6: bunlar hash'te ZATEN vardı ama parametrize listesinde yoktu,
        # yani "testle kilitli" iddiası ikisi için fazlaydı. Artık kilitli.
        ("_WORD", re.compile(r"[a-z]+")),
        ("_TITLE_LINE", re.compile(r"^X+$")),
        # review M2: gövde literalinden modül sabitine terfi eden ikisi.
        ("_TITLE_KEYWORDS", ("KANUN",)),
        ("_TR_LOWER_MAP", (("İ", "i"),)),
    ],
)
def test_recipe_fingerprint_changes_with_every_behaviour_bearing_constant(
    monkeypatch, const, value
):
    """Parmak izi reçeteyi KAPSAMALI: bir sabit değişirse anahtar da değişir.

    Bu, P2 kalibrasyon artefaktının sessizce yanlış kalmasını engelleyen
    mekanizmadır — `index_revision` bu eksenlerin hiçbirini görmez.
    """
    import belge_gozu.retrieval.text as text_mod

    before = recipe_fingerprint()
    monkeypatch.setattr(text_mod, const, value)
    assert recipe_fingerprint() != before


def test_recipe_fingerprint_covers_the_fold_table(monkeypatch):
    import belge_gozu.retrieval.text as text_mod

    before = recipe_fingerprint()
    monkeypatch.setattr(text_mod, "_FOLD", str.maketrans("ç", "c"))
    assert recipe_fingerprint() != before


def test_title_keywords_and_tr_lower_map_still_drive_behaviour(monkeypatch):
    """M2'nin gerekçesi: bu iki sabit yalnız hash'te değil, DAVRANIŞTA da yaşıyor.

    Sabite terfi ettirmek onları parmak izine soktu; bu test terfinin
    kozmetik olmadığını, gerçekten aynı kod yolunu beslediğini gösterir.
    """
    import belge_gozu.retrieval.text as text_mod

    ids, texts = ["k1:1"], ["MEDENİ KANUNU\n"]
    assert extract_doc_name_tokens(ids, texts) == {"k1": frozenset({"meden"})}
    # kapı kelimesi kaldırılınca başlık artık aday değil -> yönlendirme kapsamı boşalır
    monkeypatch.setattr(text_mod, "_TITLE_KEYWORDS", ("ANAYASA",))
    assert extract_doc_name_tokens(ids, texts) == {}

    monkeypatch.undo()
    assert tr_lower("İSTANBUL IRAK") == "istanbul ırak"
    # eşleme boşaltılınca Python'un kendi lower()'ına düşer (I -> i, İ -> i̇)
    monkeypatch.setattr(text_mod, "_TR_LOWER_MAP", ())
    assert tr_lower("IRAK") == "irak"


def test_rank_order_is_the_stable_descending_contract():
    """Sıralama ifadesi tek yerde (m11): azalan, beraberlikte indeks sırası korunur."""
    scores = np.array([1.0, 3.0, 3.0, 2.0], dtype=np.float32)
    assert rank_order(scores).tolist() == [1, 2, 3, 0]  # 3.0'lar arasında 1 önce
    np.testing.assert_array_equal(rank_order(scores), np.argsort(-scores, kind="stable"))
