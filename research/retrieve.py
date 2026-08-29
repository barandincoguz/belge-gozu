"""DENEY DOSYASI — autoresearch döngüsünün tek değiştirilebilir yüzeyi.

Sözleşme (research/program.md): rank_pages(q) -> sıralı page_id listesi.
q: query_text, page_ids, visual_scores (float32[n]), page_texts.

EXP-5 (bm25-f5-stop): exp3 + sabit Türkçe işlev-kelimesi listesi (iki tarafta:
indeks + sorgu). Liste standart dilbilgisi işlev kelimeleri — canary'ye
AYARLANMADI; içerikle çakışabilenler ("zaman" → zamanaşımı, "iş") bilinçli
dışarıda. Tek değişken: tokenizasyon filtresi.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)
F5 = 5  # Türkçe ön-ek kırpma uzunluğu

# Standart Türkçe işlev kelimeleri (tam-kelime, kırpmadan ÖNCE uygulanır).
STOPWORDS = frozenset(
    "ve veya ile için gibi göre kadar sonra önce bir bu şu o ne nasıl neden "
    "niçin hangi kaç kim mi mı mu mü midir mıdır mudur müdür da de ki en çok "
    "az her ise olan olarak üzere ancak ama fakat yoksa değil nedir sayılı".split()
)


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def tokenize(s: str) -> list[str]:
    words = [t for t in _WORD.findall(tr_lower(s)) if len(t) > 1 and t not in STOPWORDS]
    return [t[:F5] for t in words]


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = [Counter(toks) for toks in docs_tokens]
        self.doc_lens = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) or 1.0
        df: Counter[str] = Counter()
        for toks in docs_tokens:
            df.update(set(toks))
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.doc_freqs), dtype=np.float32)
        for tok in tokenize(query):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(tok)
                if f:
                    dl = self.doc_lens[i]
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out


_BM25: BM25 | None = None


def _bm25(page_texts: list[str]) -> BM25:
    global _BM25
    if _BM25 is None:
        _BM25 = BM25([tokenize(t) for t in page_texts])
    return _BM25


def rank_pages(q) -> list[str]:
    scores = _bm25(q.page_texts).scores(q.query_text)
    order = np.argsort(-scores, kind="stable")
    return [q.page_ids[i] for i in order]
