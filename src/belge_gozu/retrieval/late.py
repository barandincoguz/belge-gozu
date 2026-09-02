"""Geç-etkileşim (ColBERT) getirim kanalı — BM25'in YANINA, yerine değil.

CERRAHİ SINIR. Bu modül hiçbir mevcut dosyanın davranışını değiştirmez.
`retrieval/text.py` (BM25 reçetesi), `retrieval/hybrid.py` (sıralama) ve
`answer/base.py` (çekimserlik kapısı) olduğu gibi kalır. Kanal SAYFA
kimlikleri üretir ve `retrieval/union.py` ile mevcut sıralamanın yanına örülür.

BM25 NEDEN SAYFADA KALIYOR. Ölçüldü (insan-doğrulanmış n=47): BM25'i chunk'a
taşımak hiçbir şey kazandırmıyor.

    kol                       R@5     R@20     R@50   paraphrase R@50
    üretim bugün           0,6277   0,7660   0,8085          0,5714
    BM25 SAYFA + ColBERT   0,7766   0,9149   0,9362          0,8571
    BM25 chunk + ColBERT   0,7660   0,9149   0,9362          0,8571

Sayfa sürümü R@5'te DAHA İYİ ve dilim bazında birebir. Yani donmuş reçeteye,
`recipe_fingerprint`e ve `min_score_threshold` ölçeğine dokunmadan tüm kazanç
alınıyor — en riskli üç değişiklik gereksiz çıktı.

KAPI TEHLİKESİ VE NEDEN BAYRAK ARKASINDA. `answer/base.py` çekimserliği
`hits[0].score < min_score` ile veriyor ve o eşik (10.6) BM25 ölçeğinde
kalibre edildi. ColBERT'in bulduğu bir sayfa top-1'e girdiğinde taşıdığı BM25
skoru düşüktür: kapı YANLIŞ sebeple kapanır ve sistem bulduğu cevabı
"bulamadım" diye reddeder. Kanalın kendi kalibre eşiği olmadan bu düzeltilemez,
o yüzden `require_calibrated_late_channel` kalibrasyon artefaktı yokken açılmayı
REDDEDER. Sessizce yanlış davranmaktansa gürültüyle durmak — projenin eşik
korkuluklarının zaten izlediği desen (`app/main.py`, ruling R19).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class LateChannelNotCalibrated(RuntimeError):
    """Kanal açılmak isteniyor ama çekimserlik eşiği bu ölçekte kalibre değil."""


class QueryEncoder(Protocol):
    def encode_query_vectors(self, text: str) -> np.ndarray:
        """(n_query_token, dim) — L2 normalize edilmiş sorgu vektörleri."""


def validate_index_shapes(
    embeddings: np.ndarray, offsets: np.ndarray, chunk_ids: Sequence[str]
) -> None:
    """Artefakt kendi içinde tutarlı mı — sessiz hizasızlık en tehlikeli hatadır.

    Gömme dizisi ile chunk kimlikleri pozisyona göre hizalıdır; biri kayarsa
    getirim yanlış chunk'ları döndürür ve HİÇBİR hata vermez.
    """
    if len(chunk_ids) != len(offsets) - 1:
        raise ValueError(
            f"chunk_ids ({len(chunk_ids)}) ile ofset sayısı ({len(offsets) - 1}) uyuşmuyor"
        )
    if int(offsets[-1]) != embeddings.shape[0]:
        raise ValueError(
            f"son ofset {int(offsets[-1])} vektör sayısına ({embeddings.shape[0]}) eşit değil"
        )
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(
            "chunk_ids benzersiz değil — `{chunk_id: page_ids}` sözlüğü sayfa düşürür "
            "(gerçek veride `rg1935a:m1` 21 kez geçmiş ve 22 sayfayı erişilemez yapmıştı)"
        )


def chunk_ranking_to_pages(
    ranking: Sequence[str], chunk_pages: Mapping[str, tuple[str, ...]]
) -> list[str]:
    """Chunk sırası -> sayfa sırası; ilk görülme kazanır, tekrar atılır.

    "Getirim için chunk, kanıt için sayfa" sözleşmesi: bench'in altın verisi
    sayfa bazlıdır ve VLM cevaplayıcı sayfa görüntüsü okur.
    """
    seen: set[str] = set()
    out: list[str] = []
    for cid in ranking:
        for pid in chunk_pages.get(cid, ()):
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out


@dataclass(frozen=True)
class LateSearchResult:
    """Tek kodlamadan türeyen adaylar ve kalibrasyon istatistikleri."""

    pages: tuple[str, ...]
    query_tokens: int
    raw_top1: float
    raw_margin: float
    mean_top1: float
    mean_margin: float


@dataclass
class LateInteractionChannel:
    """MaxSim ile chunk sıralar; sayfa kimlikleri döndürür."""

    embeddings: np.ndarray          # (n_vector, dim) — fp16 diskte, fp32 skorlamada
    offsets: np.ndarray             # (n_chunk + 1,) chunk sınırları
    chunk_ids: Sequence[str]
    chunk_pages: Mapping[str, tuple[str, ...]]
    encoder: QueryEncoder

    def __post_init__(self) -> None:
        validate_index_shapes(self.embeddings, self.offsets, self.chunk_ids)

    def _score_query(self, query: str) -> tuple[np.ndarray, int]:
        """Chunk skorları + etkin sorgu tokenı sayısı; kodlamanın tek kaynağı."""
        q = np.asarray(self.encoder.encode_query_vectors(query), dtype=np.float32)
        if q.ndim != 2 or q.shape[0] == 0:
            raise ValueError(
                "sorgu vektörü matrisi boş olamaz ve (n_token, dim) biçiminde olmalı: "
                f"{q.shape}"
            )
        if q.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                "sorgu vektörü boyutu indeksle eşleşmeli: "
                f"{q.shape[1]} != {self.embeddings.shape[1]}"
            )
        sims = q @ np.asarray(self.embeddings, dtype=np.float32).T
        scores = np.maximum.reduceat(sims, self.offsets[:-1], axis=1).sum(axis=0)
        return scores, int(q.shape[0])

    def scores(self, query: str) -> np.ndarray:
        """Chunk başına MaxSim skoru — sorgu token'ları üzerinde TOPLAM."""
        return self._score_query(query)[0]

    def rank_chunks(self, query: str) -> list[str]:
        order = np.argsort(self.scores(query), kind="stable")[::-1]
        return [self.chunk_ids[i] for i in order]

    def candidate_pages(self, query: str, limit: int = 200) -> list[str]:
        """İlk `limit` chunk'ın sayfaları, sırayı koruyarak, tekrarsız."""
        return list(self.search_with_scores(query, limit=limit).pages)

    def search_with_scores(self, query: str, limit: int = 200) -> LateSearchResult:
        """Aday sayfaları ve kalibrasyon skorlarını tek sorgu kodlamasıyla üretir.

        MaxSim chunk üzerinde hesaplanır; güven marjı ise servis sözleşmesiyle
        aynı birimde, birbirinden farklı ilk iki SAYFA üzerinden ölçülür. Aynı
        sayfayı taşıyan ikinci bir chunk sahte bir top-2 rakibi sayılmaz.

        `mean_*` alanları sorgu başına sabit bir bölme olduğundan sıralamayı
        değiştirmez. Yalnız ham MaxSim toplamının etkin sorgu tokenı sayısıyla
        büyümesini çekimserlik ekseninden çıkarır.
        """
        scores, query_tokens = self._score_query(query)
        order = np.argsort(scores, kind="stable")[::-1]
        ranked_chunks = [self.chunk_ids[int(i)] for i in order]
        pages = tuple(chunk_ranking_to_pages(ranked_chunks[:limit], self.chunk_pages))

        page_scores: list[float] = []
        seen_pages: set[str] = set()
        for idx, chunk_id in zip(order, ranked_chunks, strict=True):
            for page_id in self.chunk_pages.get(chunk_id, ()):
                if page_id in seen_pages:
                    continue
                seen_pages.add(page_id)
                page_scores.append(float(scores[int(idx)]))
            if len(page_scores) >= 2:
                break

        raw_top1 = page_scores[0] if page_scores else 0.0
        raw_margin = raw_top1 - page_scores[1] if len(page_scores) >= 2 else 0.0
        return LateSearchResult(
            pages=pages,
            query_tokens=query_tokens,
            raw_top1=raw_top1,
            raw_margin=raw_margin,
            mean_top1=raw_top1 / query_tokens,
            mean_margin=raw_margin / query_tokens,
        )


def require_calibrated_late_channel(
    enabled: bool, calibrated_threshold: float | None
) -> None:
    """Kanal açıksa kalibre bir çekimserlik eşiği ZORUNLUDUR.

    Gerekçe modül başlığında: `min_score_threshold` BM25 ölçeğinde kalibre
    edildi ve ColBERT'in bulduğu bir sayfa top-1'e girdiğinde o eşik yanlış
    sebeple kapanır. Kalibrasyon yapılana kadar açılmak, ölçülen kazancı
    üretimde ÇEKİMSERLİĞE çevirebilir — sessiz ve ölçüm ortamında görünmez.
    """
    if enabled and calibrated_threshold is None:
        raise LateChannelNotCalibrated(
            "geç-etkileşim kanalı açık ama bu ölçekte kalibre bir çekimserlik eşiği yok. "
            "`min_score_threshold` BM25 ölçeğinde (bant ~4-70) kalibre edildi; ColBERT "
            "adayları o ölçekte düşük skorlar ve kapı yanlış sebeple kapanır. "
            "Kanalı açmadan önce eşiği bu kolda yeniden ölçün."
        )
