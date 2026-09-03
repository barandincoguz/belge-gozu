"""Aday havuzu için offline, skor-füzyonsuz rerank karşılaştırması."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

BGE_RERANKER_REPO = "BAAI/bge-reranker-v2-m3"
BGE_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class PageReranker(Protocol):
    """Sorgu-belge çiftlerine hizalı ham relevance skorları üretir."""

    def score(self, query: str, documents: list[str]) -> np.ndarray: ...


class TransformerPageReranker:
    """Sabit bir Hugging Face cross-encoder ile sayfa çiftlerini skorlar."""

    def __init__(
        self,
        *,
        repo: str = BGE_RERANKER_REPO,
        revision: str = BGE_RERANKER_REVISION,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size en az 1 olmalı")
        if max_length < 1:
            raise ValueError("max_length en az 1 olmalı")

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.repo = repo
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self._torch = torch
        self._device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._tokenizer: Any = AutoTokenizer.from_pretrained(repo, revision=revision)
        self._model: Any = AutoModelForSequenceClassification.from_pretrained(
            repo, revision=revision
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
) -> RerankComparison:
    """Aynı havuzun P (BM25 sabit) ve U (tam serbest) sıralamasını üretir."""
    pages = list(pool)
    documents = _require_aligned_pages(pages, page_texts, bm25_scores)
    scores = _validate_scores(reranker.score(query, documents), len(pages))
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
