"""Aday havuzu için offline, skor-füzyonsuz rerank karşılaştırması."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

BGE_RERANKER_REPO = "BAAI/bge-reranker-v2-m3"
BGE_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class PageReranker(Protocol):
    """Sorgu-belge çiftlerine hizalı ham relevance skorları üretir."""

    def score(self, query: str, documents: list[str]) -> np.ndarray: ...


class TransformerPageReranker:
    """Sabit bir Hugging Face cross-encoder ile sayfa çiftlerini skorlar.

    ``max_length`` 2026-09-11'de 512'den 4096'ya çıkarıldı. 512 sessiz bir
    ÖLÇÜM HATASIYDI: korpusun sayfa token uzunluğu medyan 617, p99 1.886, maks
    4.022 — yani sayfaların **%80,36'sı kesiliyordu** ve ilgili madde metnin
    ikinci yarısındaysa cross-encoder onu hiç görmüyordu (c203: BM25 havuzda
    2. sıraya koyuyor, reranker 40'a itiyor). 4096'da kesilme %0,00.

    Bedeli yalnız gerçekten uzun sayfalar öder: tokenizer ``padding=True`` ile
    DİNAMİK dolgu yapar, yani tavanı yükseltmek kısa girdilerin maliyetini
    değiştirmez; batch her zaman kendi en uzun örneğine kadar dolar.

    Karar (proje sahibi, 2026-09-11): gecikme yerine doğruluk tercih edilir;
    kalite kazancı ölçülebilir olduğu sürece gecikme veto değildir.
    """

    def __init__(
        self,
        *,
        repo: str = BGE_RERANKER_REPO,
        revision: str = BGE_RERANKER_REVISION,
        device: str | None = None,
        max_length: int = 4096,
        batch_size: int = 8,
        trust_remote_code: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size en az 1 olmalı")
        if max_length < 1:
            raise ValueError("max_length en az 1 olmalı")

        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.repo = repo
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self._torch = torch
        self._device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        # `trust_remote_code` uzak depodaki kodu ÇALIŞTIRIR. Proje sahibi 2026-09-11'de
        # açtı; güvenlik buradaki tek şeye bağlı: revizyon 40 karakterlik commit
        # SHA'sına PİNLİ olmalı. `main`den yüklemek, çalıştırılan kodun altımızda
        # değişmesi demektir — pin disiplininin tam olarak engellediği şey.
        if trust_remote_code and len(revision) != 40:
            raise ValueError(
                "trust_remote_code yalnız 40 karakterlik commit SHA'sıyla kullanılabilir: "
                f"{repo}@{revision}"
            )
        self.trust_remote_code = trust_remote_code
        self._tokenizer: Any = AutoTokenizer.from_pretrained(
            repo, revision=revision, trust_remote_code=trust_remote_code
        )
        self._model: Any = AutoModelForSequenceClassification.from_pretrained(
            repo, revision=revision, trust_remote_code=trust_remote_code
        ).to(self._device)
        self._model.eval()

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float64)

        batches: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(documents), self.batch_size):
                batch = documents[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [(query, document) for document in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {name: value.to(self._device) for name, value in encoded.items()}
                logits = self._model(**inputs).logits.squeeze(-1)
                if logits.ndim != 1 or logits.shape[0] != len(batch):
                    raise ValueError("transformer reranker beklenmeyen logit şekli döndürdü")
                batches.append(logits.detach().cpu().numpy())
        return np.concatenate(batches).astype(np.float64, copy=False)


@dataclass(frozen=True)
class RerankComparison:
    """BM25 birincisini sabitleyen ve serbest bırakan iki offline sıralama."""

    pinned_pages: tuple[str, ...]
    unpinned_pages: tuple[str, ...]
    bm25_top1_rank_unpinned: int
    unpinned_top1_bm25_score: float
    would_abstain: bool


def _require_aligned_pages(
    pool: Sequence[str], page_texts: Mapping[str, str], bm25_scores: Mapping[str, float]
) -> list[str]:
    if not pool:
        raise ValueError("rerank aday havuzu boş olamaz")
    docs: list[str] = []
    for page_id in pool:
        if page_id not in page_texts:
            raise ValueError(f"rerank için sayfa metni yok: {page_id}")
        if page_id not in bm25_scores:
            raise ValueError(f"rerank için BM25 skoru yok: {page_id}")
        docs.append(page_texts[page_id])
    return docs


def _validate_scores(values: np.ndarray, expected: int) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.shape != (expected,):
        raise ValueError(
            f"reranker skorları aday havuzuyla hizalı olmalı: {scores.shape} != ({expected},)"
        )
    if not np.isfinite(scores).all():
        raise ValueError("reranker skorları sonlu olmalı")
    return scores


def compare_rerankings(
    query: str,
    pool: Sequence[str],
    page_texts: Mapping[str, str],
    bm25_scores: Mapping[str, float],
    reranker: PageReranker,
    *,
    threshold: float = 10.6,
    page_scorer: Callable[[str, Sequence[str]], np.ndarray] | None = None,
) -> RerankComparison:
    """Aynı havuzun P (BM25 sabit) ve U (tam serbest) sıralamasını üretir.

    ``page_scorer`` verilirse skorlama birimi sayfa metni DEĞİLDİR: çağrılan
    fonksiyon sayfa kimliklerini alır ve kendi birimiyle (ör. madde chunk'ları,
    MaxP) sayfa skoru üretir. Hizalama guard'ı yine koşar — skorlama birimi
    değişse de havuzun sayfa/metin/BM25 üçlüsü hizalı olmak zorundadır.
    """
    pages = list(pool)
    documents = _require_aligned_pages(pages, page_texts, bm25_scores)
    raw = reranker.score(query, documents) if page_scorer is None else page_scorer(query, pages)
    scores = _validate_scores(raw, len(pages))
    unpinned = tuple(pages[int(i)] for i in np.argsort(-scores, kind="stable"))
    bm25_top1 = pages[0]
    pinned = (bm25_top1, *(page_id for page_id in unpinned if page_id != bm25_top1))
    top_score = float(bm25_scores[unpinned[0]])
    return RerankComparison(
        pinned_pages=pinned,
        unpinned_pages=unpinned,
        bm25_top1_rank_unpinned=unpinned.index(bm25_top1) + 1,
        unpinned_top1_bm25_score=top_score,
        would_abstain=top_score < threshold,
    )
