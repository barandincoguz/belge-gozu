"""Metin kanalı: Türkçe-uyarlı BM25 + doküman-adı pencere-içi yönlendirmesi.

autoresearch exp7 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md.
Bu modül `research/retrieve.py` @ 1a0624e'nin ÜRETİM PORTUDUR: sabitler
(STOPWORDS, F5=5, k1=1.5, b=0.75, WINDOW=20, _GENERIC, _TITLE_LINE) ve
tokenleştirme mantığı birebir taşınmıştır. Reçete ölçülmüş bir bütündür —
tek tek parçalar "iyileştirilirse" ölçüm geçersiz olur:

| deney | R@5 | karar |
|---|---|---|
| taban: yalnız görsel (int8) | 0.2326 | — |
| + BM25 (PDF metin katmanı) | 0.6744 | KEPT |
| eşit-ağırlık RRF(görsel, BM25) | 0.3953 | DISCARDED (zayıf kanal zarar veriyor) |
| + F5 ön-ek kırpması | 0.7674 | KEPT |
| + bigram shingle | 0.6279 | DISCARDED |
| + sabit işlev-kelime listesi | 0.7674 | KEPT (R@20 0.884->0.907, MRR+) |
| + mutlak doküman-adı bölümleme | 0.7907 | DISCARDED (R@20 gerilemesi, veto) |
| + pencere-içi (top-20) yönlendirme | **0.8140** | KEPT |

Görsel kanal F5'ten sonra top-5'e BENZERSİZ hiçbir soru katmıyor (bulgu 3),
ama serve'de telemetri/gösterim ve P2 kalibrasyon verisi için KORUNUYOR —
bkz. `retrieval/hybrid.py`.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)

# Türkçe ön-ek kırpma uzunluğu (Can vd., Turkish IR): eklemeli dilde gövde
# yaklaşık ilk 5 harftir. Ölçüm: R@5 0.6744 -> 0.7674 (+9.3 puan).
F5 = 5

# Standart Türkçe işlev kelimeleri (tam-kelime, kırpmadan ÖNCE uygulanır).
# CANARY'YE AYARLI DEĞİL: "zaman"/"iş" gibi içerik taşıyabilecek kelimeler
# bilinçli olarak dışarıda bırakıldı. Ölçüm: R@5 eşit kalır, R@20 0.884->0.907
# ve vitrin sorgusu chip1 8->4 (soru-kalıbı kelimeleri en çok ORTA sıralarda
# karışıklık üretiyor).
STOPWORDS = frozenset(
    "ve veya ile için gibi göre kadar sonra önce bir bu şu o ne nasıl neden "
    "niçin hangi kaç kim mi mı mu mü midir mıdır mudur müdür da de ki en çok "
    "az her ise olan olarak üzere ancak ama fakat yoksa değil nedir sayılı".split()
)

# Pencere = 20, R@20 guardrail'iyle BİLİNÇLİ hizalı (veriye ayar değil):
# yeniden sıralama yalnız pencerenin İÇİNDE yapıldığı için pencere KÜMESİ
# yapısal olarak değişmez, yani R@20 tanım gereği korunur.
WINDOW = 20

# Doküman adından atılan jenerik parçalar (F5-kırpık): bunlar tek başına
# kalırsa neredeyse her sorgu her kanunu "yönlendirir".
_GENERIC = frozenset({"kanun", "türk", "türki", "cumhu"})
# 1. sayfadaki büyük-harfli başlık satırı (elle doküman-adı tablosu YOK ->
# canary'den sızıntı yok; ad korpusun kendisinden türetiliyor).
_TITLE_LINE = re.compile(r"^[A-ZÇĞİÖŞÜÂÎÛ0-9 ()'’.,;:-]{8,}$")


def tr_lower(s: str) -> str:
    """Türkçe küçültme: İ->i, I->ı (Python'un `lower()`ı ikisini de 'i' yapar).

    autoresearch exp7 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md.
    """
    return s.replace("İ", "i").replace("I", "ı").lower()


def tokenize(s: str) -> list[str]:
    """`\\w+` -> tr_lower -> >=2 harf -> stopword eleme -> F5 ön-ek kırpması.

    Sıra önemlidir: stopword listesi TAM KELİME üzerinde, kırpmadan ÖNCE
    uygulanır (kırpma sonrası "göre" ve "görev" aynı token'a düşerdi).

    autoresearch exp7 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md.
    """
    words = [t for t in _WORD.findall(tr_lower(s)) if len(t) > 1 and t not in STOPWORDS]
    return [t[:F5] for t in words]


class BM25Index:
    """Sayfa metinleri üzerinde BM25 (k1=1.5, b=0.75) — `page_ids` ile hizalı.

    `research/retrieve.py`'deki `BM25` sınıfının portu (adı üretimde
    BM25Index); skorlama ifadesi birebir aynıdır.

    autoresearch exp7 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md.
    """

    def __init__(
        self, page_ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75
    ) -> None:
        if len(page_ids) != len(texts):
            raise ValueError(f"page_ids ({len(page_ids)}) ve texts ({len(texts)}) eşleşmiyor")
        if not page_ids:
            raise ValueError("boş korpus: en az bir sayfa metni gerekli")
        self.page_ids = list(page_ids)
        self.k1, self.b = k1, b
        docs_tokens = [tokenize(t) for t in texts]
        self.doc_freqs = [Counter(toks) for toks in docs_tokens]
        self.doc_lens = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) or 1.0
        df: Counter[str] = Counter()
        for toks in docs_tokens:
            df.update(set(toks))
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        """(n_pages,) float32 BM25 skorları — `page_ids` sırasıyla hizalı.

        ÖLÇEK: kalibre edilmemiş, ÜST SINIRSIZ BM25 birimi. Ölçülen bant
        (canary, 4222 sayfa): cevaplanabilir top-1'ler min 10.53 / medyan
        26.05 / maks 69.30. Görsel kanalın normalize [-1,1] skorlarıyla AYNI
        ŞEY DEĞİLDİR — eşik bu ölçekte taşınmıştır (bkz. config.py).
        """
        out = np.zeros(len(self.doc_freqs), dtype=np.float32)
        for tok in tokenize(query):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(tok)
                if f:
                    dl = self.doc_lens[i]
                    out[i] += (
                        idf
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    )
        return out


def extract_doc_name_tokens(page_ids: list[str], texts: list[str]) -> dict[str, frozenset[str]]:
    """doc_id -> 1. sayfa başlık satırından türetilmiş JENERİK-DIŞI ad token'ları.

    Adayı 1. sayfanın "KANUN"/"ANAYASA" geçen büyük-harfli satırlarından en
    UZUNU olarak seçer, tokenleştirir ve `_GENERIC`'i çıkarır. Elle yazılmış
    doküman-adı tablosu YOKTUR: ad korpusun kendisinden gelir, bu yüzden
    canary'ye ayar (sızıntı) riski yoktur.

    Adı çıkarılamayan doküman sözlükte yer almaz — yönlendirme onu hiç
    tetiklemez (kapsam dışı; ölçülen ıskalar: c206 KVKK kısaltması, c209
    Anayasa kısmi ad).

    autoresearch exp7 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md.
    """
    names: dict[str, frozenset[str]] = {}
    for pid, text in zip(page_ids, texts, strict=True):
        doc, _, page = pid.partition(":")
        if page != "1":
            continue
        cands = [
            ln.strip()
            for ln in text.splitlines()
            if ("KANUN" in ln or "ANAYASA" in ln) and _TITLE_LINE.match(ln.strip())
        ]
        if not cands:
            continue
        toks = frozenset(tokenize(max(cands, key=len))) - _GENERIC
        if toks:
            names[doc] = toks
    return names


def route_window(ranking: list[str], routed_docs: set[str], window: int = WINDOW) -> list[str]:
    """İlk `window` girdiyi YENİDEN SIRALAR: yönlendirilen dokümanların sayfaları öne.

    Sözleşme (yapısal, ölçüme bağlı değil):
      * ilk `window` girdinin KÜMESİ değişmez (yalnız kendi içinde sıralanır),
        bu yüzden R@window guardrail'i tanım gereği korunur;
      * `window`'dan sonrası hiç dokunulmaz;
      * her iki grup içinde göreli sıra (BM25 sırası) korunur.

    exp6'nın MUTLAK bölümlemesi (aday kümesini değiştiren sürüm) tam da bu
    sözleşmeyi deldiği için veto edildi: R@5 0.7907'ye çıkarken R@20
    0.907->0.837'ye geriliyordu. Pencere-içi sürüm R@5 0.8140 + R@20 0.907.

    autoresearch exp7 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md.
    """
    if not routed_docs:
        return list(ranking)
    win = ranking[:window]
    front = [pid for pid in win if pid.partition(":")[0] in routed_docs]
    back = [pid for pid in win if pid.partition(":")[0] not in routed_docs]
    return front + back + list(ranking[window:])
