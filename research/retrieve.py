"""DENEY DOSYASI — autoresearch döngüsünün tek değiştirilebilir yüzeyi.

Sözleşme (research/program.md): rank_pages(q) -> sıralı page_id listesi.
q: query_text, page_ids, visual_scores (float32[n]), page_texts.

EXP-12 (ascii-fold, KEPT): exp8 + tokenizasyonda aksan katlama — tr_lower
SONRASI ç→c, ğ→g, ı→i, ö→o, ş→s, ü→u, â→a, î→i, û→u; iki tarafta (indeks+sorgu);
stopword listesi katlanmış biçimiyle eşlenir; F5 katlamadan SONRA. Gerekçe: aksansız
yazan gerçek kullanıcıda R@5 0.837→0.581 çöküyordu (exp11); katlama sistemi
YAZIM-DEĞİŞMEZ yapar: iki koşulda da 37/43=0.8605. Bedel: aksanlı MRR 0.655→0.632
(çakışma kaynaklı, düşenler top-5 İÇİNDE kalır) — program round-3 R26 istisnasıyla
kabul. exp13 (çift-biçim) denendi ve iki guardrail'i düşürdüğü için reddedildi.

EXP-14 (qtf≤2 sorgu doygunluğu, KEPT): `BM25.scores` artık sorgunun BENZERSİZ
token'ları üzerinde gezip her terime `min(qtf, 2)` ağırlığı veriyor. Reçete
değişikliği ÜRETİMDEN geldi (bulgu Y1): canary'de hiçbir sorgu terim tekrar
etmediği için ölçüm uzayı bu sınıfı hiç görmemişti, üretimde ise "ihbar"×80
skoru 667.5'e çıkarıp eşiği anlamsızlaştırıyordu. Ölçüm ledger 2026-08-30.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)
F5 = 5  # Türkçe ön-ek kırpma uzunluğu
QTF_CAP = 2  # exp14: sorgu-terim doygunluk tavanı (üretim portu: retrieval/text.py)

# Standart Türkçe işlev kelimeleri (tam-kelime, kırpmadan ÖNCE uygulanır).
STOPWORDS = frozenset(
    "ve veya ile için gibi göre kadar sonra önce bir bu şu o ne nasıl neden "
    "niçin hangi kaç kim mi mı mu mü midir mıdır mudur müdür da de ki en çok "
    "az her ise olan olarak üzere ancak ama fakat yoksa değil nedir sayılı".split()
)


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def ascii_fold(s: str) -> str:
    return s.translate(_FOLD)


_STOP_FOLDED = frozenset(ascii_fold(w) for w in STOPWORDS)


def tokenize(s: str) -> list[str]:
    words = [ascii_fold(t) for t in _WORD.findall(tr_lower(s)) if len(t) > 1]
    return [t[:F5] for t in words if t not in _STOP_FOLDED]


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
        """BM25 skorları; sorgu BENZERSİZ token'lar üzerinde `min(qtf, QTF_CAP)` ağırlıkla.

        qtf≤2 doygunluğu — canary birebir değişmedi: R@5 37/43, MRR 0.6320;
        saldırı 'ihbar'×80 top-1 667.5→16.7; ölçüm ledger 2026-08-30.
        """
        out = np.zeros(len(self.doc_freqs), dtype=np.float32)
        for tok, qtf in Counter(tokenize(query)).items():
            idf = self.idf.get(tok)
            if idf is None:
                continue
            qw = min(qtf, QTF_CAP)
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(tok)
                if f:
                    dl = self.doc_lens[i]
                    norm = 1 - self.b + self.b * dl / self.avgdl
                    out[i] += qw * idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out


_BM25: BM25 | None = None


def _bm25(page_texts: list[str]) -> BM25:
    global _BM25
    if _BM25 is None:
        _BM25 = BM25([tokenize(t) for t in page_texts])
    return _BM25


WINDOW = 50  # exp8: guardrail artık yapısal değil ölçülü (bkz. docstring)

_GENERIC = frozenset({"kanun", "türk", "türki", "cumhu"})
_TITLE_LINE = re.compile(r"^[A-ZÇĞİÖŞÜÂÎÛ0-9 ()'’.,;:-]{8,}$")
_DOC_NAMES: dict[str, frozenset[str]] | None = None


def _doc_name_tokens(page_ids: list[str], page_texts: list[str]) -> dict[str, frozenset[str]]:
    """doc_id -> 1. sayfadaki başlık satırından türetilmiş jenerik-dışı ad token'ları."""
    global _DOC_NAMES
    if _DOC_NAMES is not None:
        return _DOC_NAMES
    names: dict[str, frozenset[str]] = {}
    for pid, text in zip(page_ids, page_texts, strict=True):
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
    _DOC_NAMES = names
    return names


def rank_pages(q) -> list[str]:
    scores = _bm25(q.page_texts).scores(q.query_text)
    order = np.argsort(-scores, kind="stable")
    ranking = [q.page_ids[i] for i in order]

    q_toks = set(tokenize(q.query_text))
    names = _doc_name_tokens(q.page_ids, q.page_texts)
    routed = {doc for doc, toks in names.items() if toks <= q_toks}
    if not routed:
        return ranking
    win = ranking[:WINDOW]
    front = [pid for pid in win if pid.partition(":")[0] in routed]
    back = [pid for pid in win if pid.partition(":")[0] not in routed]
    return front + back + ranking[WINDOW:]
