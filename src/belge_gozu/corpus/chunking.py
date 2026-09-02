"""Yapı-farkında chunking: madde bazlı, sayfa yedekli.

NEDEN sayfa değil madde. 2026-09-02'de 4.060 kanun sayfası üzerinde ölçüldü:
maddelerin **%29,3'ü sayfa sınırını aşıyor**. Sayfa bazlı getirim bu maddeleri
yapısal olarak ikiye böler ve bunun canlı örneği bench'te zaten var — `c213`'te
VUK m.17'nin üçüncü şartı altın sayfa `k213:8`'de değil `k213:9`'daydı; çapraz
kontrol turu gold'a ikinci sayfayı elle eklemek zorunda kalmıştı. Madde chunk'ı
o hata sınıfını tanım gereği ortadan kaldırır.

NEDEN karışık granülerlik. Korpus tek bir yapısal rejimde değil:

  madde gövdesi      3.755 sayfa  %88,9   -> madde chunk'ı
  ek / tarife bölgesi  305 sayfa  %7,2    -> sayfa chunk'ı
  RG taramaları        162 sayfa  %3,8    -> sayfa chunk'ı (18/162'de yapı var)

Kanunların sonundaki tarife/cetvel ekleri madde değildir; onları son maddeye
yapıştırmak 112.000 karakterlik bir chunk üretiyordu (k492). Bench de bunu
doğruluyor: `tablo-layout` sorularının altın sayfaları (`k193:154`, `k193:155`)
tam olarak ek bölgesinde ve orada doğru birim sayfadır.

GETİRİM İÇİN CHUNK, KANIT İÇİN SAYFA. Chunk yalnız bir iç temsildir; her chunk
taşıdığı `page_ids`'i bilir. Skorlama ve VLM cevaplayıcı chunk -> sayfa
eşlemesiyle çalışır, böylece bench'in altın verisi sayfa bazlı KALIR ve ölçüm
sürekliliği kırılmaz.

UYARI — BM25 reçetesi. `k1=1.5, b=0.75` sayfa uzunluğunda (medyan 2.600
karakter) ölçüldü; madde chunk'ının medyanı 516. `b` tam olarak uzunluk
normalizasyonunu kontrol ettiğinden reçetenin davranışı DEĞİŞİR. Bu modül
üretime alınırken `retrieval.text.RECIPE_VERSION` artırılmalı ve parmak izine
chunk birimi girmelidir — aksi halde sayfa üzerinde eğitilmiş bir kalibratör
madde chunk'larına sessizce takılır (P2 denetimi T6 bulgusu).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# Madde başlığının TEK sabitlenmiş tanımı. Tire ZORUNLUDUR: onsuz "5 inci
# maddesi uyarınca" gibi ATIFLAR da sınır sayılırdı. Ölçümde iki varyant
# arasında %10 fark vardı (10.189 vs 11.179 madde); bu tanım dar olanıdır.
_ARTICLE = re.compile(
    r"(?:^|\n)[ \t]*"
    r"(?:(?P<kind>EK|Ek|GEÇİCİ|Geçici|GECICI|MÜKERRER|Mükerrer)[ \t]+)?"
    r"(?:MADDE|Madde)[ \t]+(?P<no>\d+)"
    r"(?:[ \t]*/[ \t]*(?P<sfx>[A-ZÇĞİÖŞÜ]))?"
    r"[ \t]*[-–—]"
)
# TUZAK: bu sözlük HAM eşleşme metniyle aranır, `.lower()` ile DEĞİL.
# Python'un locale'siz `.lower()`'ı Türkçe noktalı İ'yi birleşik noktaya çevirir
# ("GEÇİCİ".lower() != "geçici"), yani lowercase anahtarlı bir arama GEÇİCİ
# maddeleri sessizce normal madde sayardı. `retrieval.text.tr_lower` bu sınıf
# için var; burada arama zaten sonlu bir küme olduğundan biçimleri açıkça
# listelemek daha ucuz ve daha az kırılgan.
_KIND = {
    "EK": "ek", "Ek": "ek",
    "GEÇİCİ": "gecici", "Geçici": "gecici", "GECICI": "gecici",
    "MÜKERRER": "mukerrer", "Mükerrer": "mukerrer",
}

# Dipnot BAŞI: satır başında çıplak dipnot numarası + mevzuat tarihi.
# `Madde 2 – (Değişik: 2/5/2001 - 4667/4 md.)` bu desene UYMAZ (satır "Madde"
# ile başlar) ve gövdenin parçası olarak kalır — orası maddenin kendi künyesi.
_FOOTNOTE_HEAD = re.compile(r"^[ \t]*\d{1,3}[ \t]+\d{1,2}/\d{1,2}/\d{4}[ \t]+tarihli")
_HISTORY = re.compile(r"ibaresi|değiştiril|eklenmiş|yürürlükten|maddesiyle|md\.\)", re.I)

# Ek/tarife bölgesi işaretleri (kanun sonundaki cetveller).
_ANNEX = re.compile(
    r"SAYILI[ \t]+TARİFE|SAYILI[ \t]+CETVEL|SAYILI[ \t]+LİSTE|EKLİ[ \t]+CETVEL", re.I
)

# Madde rejimi için ASGARİ işaret yoğunluğu (sayfa başına madde başlığı).
# Kanunlarda ölçülen değer ~2,5; RG taramalarında 18/162 = 0,11. Aradaki
# uçurum geniş, eşik hassas değil. Yoğunluk altındaysa belge TAMAMEN sayfa
# rejimine düşer: gazete sayfalarında geçen `Madde 22-` yayımlanan kanunun
# içindedir, gazetenin kendi yapısı değildir — ve tek bir başıboş işaret
# 20 sayfayı tek chunk'a yutuyordu (156.086 karakter).
_MIN_MARKER_DENSITY = 0.3

# Bir madde chunk'ının kapsayabileceği AZAMİ sayfa sayısı. Eşik veriden
# seçildi: gerçek maddelerin %99,73'ü ≤5 sayfa (1 sayfa %72,2 · ≤2 sayfa
# %97,5). 6+ sayfaya yayılan 26 chunk'ın (%0,27) hepsi ayrıştırma hatasıydı —
# damga vergisi tarifeleri (`k488:m33`, 17 sayfa), DMK kadro cetvelleri
# (`k657:m36`, 16 sayfa) ve gazete yutmaları (`rg1975a:m22`, 20 sayfa).
_MAX_ARTICLE_PAGES = 6

_MAX_HEADING_LINES = 3
_MAX_HEADING_CHARS = 90
_NUMBERING_ONLY = re.compile(r"^[IVXAB0-9]{1,4}\.$")


@dataclass(frozen=True)
class Chunk:
    """Getirim birimi. `page_ids` tuple'dır: donuk ve hashable olsun diye."""

    chunk_id: str
    doc_id: str
    kind: Literal["article", "page"]
    heading: str
    text: str
    page_ids: tuple[str, ...]


@dataclass(frozen=True)
class ArticleMarker:
    article_id: str
    start: int


def strip_footnotes(text: str) -> str:
    """Mevzuat tarihçesi dipnotlarını gövdeden ayıklar.

    Dipnotlar karakterlerin %5,84'ünü kaplıyor ve sayfaların %38,3'ünde
    bulunuyor; üstelik PyMuPDF onları sayfa altından okuyup gövdenin ORTASINA
    sokuyor, yani cümleyi bölüyorlar. BM25 için gürültü token'ı, yoğun gömücü
    için embedding'i seyrelten dolgu.
    """
    out: list[str] = []
    dropping = False
    for line in text.split("\n"):
        if _FOOTNOTE_HEAD.match(line):
            dropping = True
            continue
        if dropping:
            # dipnot çok satırlı olabilir; tarihçe dili sürdükçe atmaya devam et
            if _HISTORY.search(line):
                continue
            dropping = False
        out.append(line)
    return "\n".join(out)


def extract_heading(lines: list[str], index: int) -> str:
    """`lines[index]` madde satırıysa onun kenar başlığı ("" olabilir).

    Başlıklar kendi satırlarındadır ve madde satırının hemen üstünde durur:

        V. Yerleşim yeri
        1. Tanım
        Madde 19- Yerleşim yeri bir kimsenin...

    Geriye doğru yürür ve gövde cümlesine çarpınca durur (nokta ile biten
    satır). Ölçüm: maddelerin %86'sında başlık yakalanıyor, medyan 31 karakter.
    Bunlar insan eliyle yazılmış konu etiketleridir — paraphrase açığına ham
    gövdeden daha iyi çalışmaları beklenir.
    """
    parts: list[str] = []
    for j in range(index - 1, max(-1, index - 1 - _MAX_HEADING_LINES), -1):
        raw = lines[j]
        s = raw.strip()
        if not s:
            break
        if _ARTICLE.match("\n" + raw):
            break
        if s.endswith(".") and not _NUMBERING_ONLY.match(s):
            break
        if len(s) > _MAX_HEADING_CHARS:
            break
        parts.append(s)
    return " / ".join(reversed(parts))


def find_article_markers(text: str) -> list[ArticleMarker]:
    """Metindeki madde sınırları, geçtikleri sırada."""
    out: list[ArticleMarker] = []
    for m in _ARTICLE.finditer(text):
        kind = _KIND.get(m.group("kind") or "", "m")
        aid = f"{kind}{m.group('no')}" + (f"/{m.group('sfx')}" if m.group("sfx") else "")
        # eşleşme baştaki \n'i de yiyor; chunk metni madde satırından başlasın
        start = m.start() + (1 if text[m.start()] == "\n" else 0)
        out.append(ArticleMarker(article_id=aid, start=start))
    return out


def annex_start_page(pages: list[tuple[int, str]]) -> int | None:
    """Ek/tarife bölgesinin başladığı sayfa numarası (yoksa None).

    Kural ölçümden geldi: madde başlığı YOK + tarife/cetvel işareti VAR. Madde
    başlığı taşıyan bir sayfa hâlâ madde rejimindedir — `Madde 2- EKLİ CETVEL
    uyarınca...` bir atıftır, bölge başlangıcı değil.
    """
    for pno, text in pages:
        if _ANNEX.search(text) and not _ARTICLE.search("\n" + text):
            return pno
    return None


def chunk_document(doc_id: str, pages: list[tuple[int, str]]) -> list[Chunk]:
    """Bir belgeyi chunk'lara böler; madde rejimi + sayfa yedeği."""
    pages = sorted(pages, key=lambda p: p[0])
    cleaned = [(pno, strip_footnotes(text)) for pno, text in pages]

    if not _has_article_regime(cleaned):
        return [_page_chunk(doc_id, pno, text) for pno, text in cleaned]

    annex_from = annex_start_page(cleaned)
    article_pages = [p for p in cleaned if annex_from is None or p[0] < annex_from]

    chunks = list(_article_chunks(doc_id, article_pages))
    if not chunks:
        # yapısı olmayan belge (RG taraması) tamamen sayfa rejimine düşer
        return [_page_chunk(doc_id, pno, text) for pno, text in cleaned]

    # KAPSAMA DEĞİŞMEZİ: hiçbir sayfa kaybolamaz. Madde chunk'larının
    # dokunmadığı her sayfa sayfa-chunk'ı olur. Bu, ilk madde başlığından
    # ÖNCEKİ metni (kapak, başlangıç hükümleri, fihrist) kurtarır — gerçek
    # korpusta 11 sayfa, `rg1975a:1` altın sayfası dahil, sessizce düşüyordu.
    # Aşırı büyük bloklar ayrıştırma hatasıdır: düşür, sayfaları aşağıdaki
    # kapsama adımı sayfa-chunk'ı olarak toplasın. Düşen bloğun sayfaları
    # başka bir madde tarafından da tutuluyor olabilir; o durumda sayfa iki
    # temsilde birden görünür — getirimde tekrar, kayıptan iyidir.
    oversized = [c for c in chunks if len(c.page_ids) >= _MAX_ARTICLE_PAGES]
    chunks = [c for c in chunks if len(c.page_ids) < _MAX_ARTICLE_PAGES]
    forced = {pid for c in oversized for pid in c.page_ids}

    touched = {pid for c in chunks for pid in c.page_ids} - forced
    uncovered = [(pno, text) for pno, text in cleaned if f"{doc_id}:{pno}" not in touched]
    chunks += [_page_chunk(doc_id, pno, text) for pno, text in uncovered]
    return sorted(chunks, key=lambda c: int(c.page_ids[0].split(":")[1]))


def _has_article_regime(pages: list[tuple[int, str]]) -> bool:
    """Belge madde bazlı chunk'lanacak kadar düzenli mi?"""
    if not pages:
        return False
    markers = sum(len(find_article_markers("\n" + text)) for _, text in pages)
    return markers / len(pages) >= _MIN_MARKER_DENSITY


def _page_chunk(doc_id: str, pno: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:p{pno}",
        doc_id=doc_id,
        kind="page",
        heading="",
        text=text,
        page_ids=(f"{doc_id}:{pno}",),
    )


def _article_chunks(doc_id: str, pages: list[tuple[int, str]]):
    if not pages:
        return
    blob_parts: list[str] = []
    spans: list[tuple[int, int, int]] = []  # (başlangıç, bitiş, sayfa no)
    pos = 0
    for pno, text in pages:
        blob_parts.append(text)
        spans.append((pos, pos + len(text), pno))
        pos += len(text) + 1
    blob = "\n".join(blob_parts)

    markers = find_article_markers(blob)
    if not markers:
        return
    lines = blob.split("\n")
    line_start = _line_start_offsets(blob)

    # Madde numarası bir belgede TEKRARLANABİLİR: bir Resmî Gazete sayısı birden
    # çok kanun yayımlar ve her birinin kendi "Madde 1-"i vardır (`rg1935a:m1`
    # gerçek veride 21 kez geçiyordu). Tekrarlı kimlik, `{chunk_id: page_ids}`
    # sözlüğünde son-kayıt-kazanır davranışıyla 22 sayfayı ERİŞİLEMEZ yapıyordu
    # — bench gold sayfası `rg1935a:1` dahil. İlk geçiş düz kimliği KORUR
    # (bench'in `gold_article_ids` alanı o biçimi kullanıyor), sonrakiler
    # sıra numarası alır.
    seen_ids: dict[str, int] = {}

    for i, mk in enumerate(markers):
        end = markers[i + 1].start if i + 1 < len(markers) else len(blob)
        body = blob[mk.start : end].strip()
        touched = tuple(f"{doc_id}:{pno}" for (a, b, pno) in spans if a < end and b > mk.start)
        n = seen_ids.get(mk.article_id, 0) + 1
        seen_ids[mk.article_id] = n
        suffix = "" if n == 1 else f"#{n}"
        yield Chunk(
            chunk_id=f"{doc_id}:{mk.article_id}{suffix}",
            doc_id=doc_id,
            kind="article",
            heading=extract_heading(lines, line_start[mk.start]),
            text=body,
            page_ids=touched,
        )


def _line_start_offsets(blob: str) -> dict[int, int]:
    """Karakter ofseti -> satır indeksi (yalnız satır BAŞLARI için)."""
    out: dict[int, int] = {}
    pos = 0
    for idx, line in enumerate(blob.split("\n")):
        out[pos] = idx
        pos += len(line) + 1
    return out
