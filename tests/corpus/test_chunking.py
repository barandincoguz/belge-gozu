"""Madde bazlı chunking — saf mantık testleri (PDF/indeks I/O YOK).

Ölçüm tabanı (2026-09-02, 4.060 kanun sayfası üzerinde):
  * maddelerin %29,3'ü sayfa sınırını aşıyor -> chunk sayfa DEĞİL madde olmalı
  * korpusun %7,2'si ek/tarife bölgesi, %3,8'i RG taraması -> oralarda yapı yok
  * dipnotlar karakterlerin %5,84'ünü kaplıyor ve gövdenin İÇİNE giriyor
  * kenar başlığı maddelerin %86'sında var (medyan 31 karakter)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from belge_gozu.corpus.chunking import (  # noqa: E402
    Chunk,
    annex_start_page,
    chunk_document,
    extract_heading,
    find_article_markers,
    strip_footnotes,
)

# --------------------------------------------------------------------------
# dipnot temizliği
# --------------------------------------------------------------------------


def test_strip_footnotes_removes_amendment_history_line():
    text = (
        "işverenler bakımından gider olarak dikkate alınmaz.\n"
        "190 2/7/2018 tarihli ve 703 sayılı Kanun Hükmünde Kararnamenin 203 üncü "
        "maddesiyle bu fıkrada yer alan ibare değiştirilmiştir.\n"
        "bir yıl süreyle bu maddeyle sağlanan destekten yararlanamaz.\n"
    )
    out = strip_footnotes(text)
    assert "703 sayılı Kanun Hükmünde" not in out
    assert "gider olarak dikkate alınmaz" in out
    assert "bir yıl süreyle" in out


def test_strip_footnotes_keeps_inline_amendment_marker_in_article_head():
    """`Madde 2 – (Değişik: 2/5/2001 - 4667/4 md.)` gövdenin PARÇASI, dipnot değil."""
    text = "Madde 2 – (Değişik: 2/5/2001 - 4667/4 md.) Avukatlık, kamu hizmetidir.\n"
    assert strip_footnotes(text) == text


def test_strip_footnotes_leaves_clean_text_untouched():
    text = "Madde 19- Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir.\n"
    assert strip_footnotes(text) == text


# --------------------------------------------------------------------------
# kenar başlığı
# --------------------------------------------------------------------------


def test_extract_heading_joins_two_level_heading():
    lines = [
        "Kayın hısımlığı, evliliğin sona ermesiyle ortadan kalkmaz.",
        "",
        "V. Yerleşim yeri",
        "1. Tanım",
        "Madde 19- Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir.",
    ]
    assert extract_heading(lines, 4) == "V. Yerleşim yeri / 1. Tanım"


def test_extract_heading_single_line():
    lines = ["Tüketicinin seçimlik hakları", "MADDE 11- (1) Malın ayıplı olduğunun..."]
    assert extract_heading(lines, 1) == "Tüketicinin seçimlik hakları"


def test_extract_heading_stops_at_body_prose():
    """Önceki satır cümleyse başlık değildir — nokta ile biter."""
    lines = ["Bu kural ticarî kuruluşlar hakkında uygulanmaz.", "Madde 20- Bir yerleşim yerinin..."]
    assert extract_heading(lines, 1) == ""


def test_extract_heading_empty_when_article_is_first_line():
    assert extract_heading(["Madde 1 – Avukatlık, kamu hizmetidir."], 0) == ""


# --------------------------------------------------------------------------
# madde başlığı bulma — TEK sabitlenmiş tanım
# --------------------------------------------------------------------------


def test_find_article_markers_handles_plain_ek_gecici_mukerrer():
    text = (
        "Madde 19- gövde\n"
        "EK MADDE 3- gövde\n"
        "GEÇİCİ MADDE 5 – gövde\n"
        "Mükerrer Madde 355- gövde\n"
    )
    assert [m.article_id for m in find_article_markers(text)] == [
        "m19", "ek3", "gecici5", "mukerrer355",
    ]


def test_find_article_markers_requires_dash_so_cross_references_are_ignored():
    """`5 inci maddesi uyarınca` bir ATIFTIR; chunk sınırı değildir."""
    text = "Madde 19- Bu Kanunun 5 inci maddesi uyarınca Madde 7 hükmü saklıdır.\n"
    assert [m.article_id for m in find_article_markers(text)] == ["m19"]


def test_find_article_markers_accepts_letter_suffix():
    assert [m.article_id for m in find_article_markers("MADDE 8/A- gövde\n")] == ["m8/A"]


# --------------------------------------------------------------------------
# ek/tarife bölgesi
# --------------------------------------------------------------------------


def test_annex_start_page_detects_tariff_region():
    pages = [
        (1, "Madde 1- gövde"),
        (2, "Madde 2- gövde"),
        (3, "(1) SAYILI TARİFE\nYargı harçları listesi"),
        (4, "devam eden tarife satırları"),
    ]
    assert annex_start_page(pages) == 3


def test_annex_start_page_none_when_no_tariff():
    pages = [(1, "Madde 1- gövde"), (2, "Madde 2- gövde")]
    assert annex_start_page(pages) is None


def test_annex_marker_on_page_that_still_has_articles_is_not_annex_start():
    """Madde başlığı taşıyan sayfa hâlâ madde rejimindedir."""
    pages = [(1, "Madde 1- gövde"), (2, "Madde 2- EKLİ CETVEL uyarınca hesaplanır")]
    assert annex_start_page(pages) is None


# --------------------------------------------------------------------------
# belge chunklama
# --------------------------------------------------------------------------


def test_chunk_document_produces_one_chunk_per_article():
    pages = [(1, "Başlık A\nMadde 1- birinci gövde.\nBaşlık B\nMadde 2- ikinci gövde.")]
    chunks = chunk_document("k1", pages)
    assert [c.chunk_id for c in chunks] == ["k1:m1", "k1:m2"]
    assert chunks[0].heading == "Başlık A"
    assert chunks[0].kind == "article"


def test_chunk_spanning_two_pages_carries_both_page_ids():
    """Maddelerin %29,3'ü sayfa sınırını aşıyor; c213 tam bu sınıftandı."""
    pages = [(8, "Madde 17- birinci şart ve ikinci şart"), (9, "üçüncü şart burada biter.")]
    chunks = chunk_document("k213", pages)
    assert len(chunks) == 1
    assert chunks[0].page_ids == ("k213:8", "k213:9")
    assert "üçüncü şart" in chunks[0].text


def test_annex_pages_become_page_chunks():
    pages = [(1, "Madde 1- gövde."), (2, "(1) SAYILI TARİFE\nsatırlar"), (3, "devam")]
    chunks = chunk_document("k492", pages)
    assert [c.chunk_id for c in chunks] == ["k492:m1", "k492:p2", "k492:p3"]
    assert chunks[1].kind == "page"


def test_document_without_article_structure_falls_back_to_page_chunks():
    """RG taramalarının 162 sayfasının 144'ünde madde işareti yok."""
    pages = [(1, "T.C. Resmî Gazete 1 HAZİRAN 1965"), (2, "ikinci sayfa metni")]
    chunks = chunk_document("rg1965a", pages)
    assert [c.chunk_id for c in chunks] == ["rg1965a:p1", "rg1965a:p2"]
    assert all(c.kind == "page" for c in chunks)


def test_chunk_text_is_footnote_free():
    pages = [
        (
            1,
            "Madde 1- gövde cümlesi.\n"
            "12 3/4/2011 tarihli ve 6111 sayılı Kanunun 5 inci maddesiyle "
            "bu fıkra değiştirilmiştir.\n"
            "gövde devam ediyor.",
        )
    ]
    text = chunk_document("k1", pages)[0].text
    assert "6111 sayılı" not in text
    assert "gövde devam ediyor" in text


def test_chunk_is_hashable_and_frozen():
    c = Chunk(chunk_id="k1:m1", doc_id="k1", kind="article", heading="H", text="T",
              page_ids=("k1:1",))
    assert hash(c)


# --------------------------------------------------------------------------
# kapsama değişmezi
#
# Gerçek korpusta koşturunca yakalandı: `rg1975a:1` altın sayfası hiçbir
# chunk'ta yoktu. İlk madde başlığından ÖNCEKİ metin (kapak, başlangıç,
# fihrist) sessizce düşüyordu — getirimin asla ulaşamayacağı sayfa demek.
# --------------------------------------------------------------------------


def test_pages_before_first_article_marker_are_not_lost():
    pages = [(1, "T.C. Resmî Gazete — kapak"), (2, "Madde 1- gövde.")]
    covered = {p for c in chunk_document("rg1975a", pages) for p in c.page_ids}
    assert "rg1975a:1" in covered


def test_every_input_page_is_covered_by_some_chunk():
    """Değişmez: hiçbir sayfa chunk'lamada kaybolamaz."""
    pages = [
        (1, "başlangıç hükümleri, madde başlığı yok"),
        (2, "Madde 1- gövde."),
        (3, "Madde 2- gövde."),
        (4, "(1) SAYILI TARİFE"),
        (5, "tarife devamı"),
    ]
    covered = {p for c in chunk_document("k1", pages) for p in c.page_ids}
    assert covered == {f"k1:{n}" for n in (1, 2, 3, 4, 5)}


def test_sparse_article_markers_fall_back_to_page_regime():
    """RG taramasında başıboş bir `Madde 22-` 20 sayfayı yutmuştu (156k karakter).

    Gazete sayfalarında geçen madde başlıkları yayımlanan kanunun İÇİNDEDİR;
    gazetenin kendi yapısı değildir. 162 RG sayfasının yalnız 18'inde işaret
    var — bu yoğunlukta madde rejimi anlamsız.
    """
    pages = [(n, f"gazete sayfası {n} metni") for n in range(1, 21)]
    pages[7] = (8, "Madde 22- yayımlanan kanunun bir maddesi")
    chunks = chunk_document("rg1975a", pages)
    assert all(c.kind == "page" for c in chunks)
    assert len(chunks) == 20


def test_oversized_article_block_falls_back_to_page_chunks():
    """6+ sayfaya yayılan "madde" bir ayrıştırma hatasıdır, madde değil.

    Ölçüm: gerçek maddelerin %99,73'ü ≤5 sayfa. 6+ sayfaya yayılan 26 chunk'ın
    (%0,27) hepsi tablo/cetvel yutması ya da RG taramasında başıboş bir işaret
    (`rg1975a:m22` tek başına 20 sayfa, 156.086 karakter).
    """
    # işaret yoğunluğu madde rejiminde kalmaya yetmeli (gerçek kanunlarda ~2,5)
    pages = [(n, f"Madde {n}- kısa gövde.") for n in range(1, 5)]
    pages.append((5, "Madde 5- uzun blok başlıyor."))
    pages += [(n, f"devam {n}") for n in range(6, 13)]
    ids = [c.chunk_id for c in chunk_document("k1", pages)]
    assert "k1:m1" in ids, "normal madde korunmalı"
    assert "k1:m5" not in ids, "8 sayfaya yayılan blok düşmeli"
    assert "k1:p12" in ids, "düşen bloğun sayfaları sayfa-chunk'ı olmalı"


# --------------------------------------------------------------------------
# chunk_id benzersizliği
#
# Gerçek veride yakalandı: bir Resmî Gazete SAYISI birden çok kanun yayımlıyor
# ve her birinin kendi "Madde 1-"i var. `rg1935a:m1` 21 kez tekrarlanıyordu.
# `{chunk_id: page_ids}` sözlüğü kurulduğunda son kayıt kazanıyor ve 22 sayfa
# erişilemez oluyordu — bench gold sayfası `rg1935a:1` dahil.
# --------------------------------------------------------------------------


def test_repeated_article_numbers_get_unique_chunk_ids():
    """Aynı belgede ikinci kez geçen madde numarası ayrı kimlik almalı."""
    pages = [
        (1, "Madde 1- birinci kanunun ilk maddesi."),
        (2, "Madde 2- birinci kanunun ikinci maddesi."),
        (3, "Madde 1- İKİNCİ kanunun ilk maddesi."),
        (4, "Madde 2- ikinci kanunun ikinci maddesi."),
    ]
    ids = [c.chunk_id for c in chunk_document("rg1935a", pages)]
    assert len(ids) == len(set(ids)), f"tekrarlı kimlik: {ids}"


def test_first_occurrence_keeps_the_plain_id():
    """Bench'in gold_article_ids'i düz biçimi kullanıyor (`k4721:m19`) — bozulmamalı."""
    pages = [(1, "Madde 1- ilk."), (2, "Madde 1- tekrar.")]
    ids = [c.chunk_id for c in chunk_document("d", pages)]
    assert ids[0] == "d:m1"
    assert ids[1] != "d:m1"


def test_chunk_id_map_reaches_every_page():
    """Değişmez: kimlikten sayfaya sözlük hiçbir sayfayı düşürmemeli."""
    pages = [(n, f"Madde 1- {n}. kanunun maddesi.") for n in range(1, 6)]
    chunks = chunk_document("rg", pages)
    cmap = {c.chunk_id: c.page_ids for c in chunks}
    reachable = {p for ps in cmap.values() for p in ps}
    assert reachable == {f"rg:{n}" for n in range(1, 6)}
