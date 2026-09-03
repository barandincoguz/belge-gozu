"""Offline semantic aday kollarını skor füzyonu yapmadan ölçer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from belge_gozu.bench.metrics import mrr, ndcg_at_k, recall_at_k
from belge_gozu.retrieval.candidates import build_candidate_pool


class CandidateChannel(Protocol):
    def candidate_pages(self, query: str, limit: int) -> list[str]: ...


class CoverageQuestion(Protocol):
    @property
    def question_id(self) -> str: ...

    @property
    def question(self) -> str: ...

    @property
    def answerable(self) -> bool: ...

    @property
    def gold_page_ids(self) -> list[str]: ...

    @property
    def slice(self) -> str: ...


def _summary(
    rows: Sequence[tuple[set[str], list[str]]],
) -> dict[str, float | int | dict[int, float]]:
    if not rows:
        raise ValueError("semantic kapsama için cevaplanabilir soru yok")
    relevant = [gold for gold, _ in rows]
    return {
        "coverage": float(
            np.mean([recall_at_k(gold, pages, len(pages)) for gold, pages in rows])
        ),
        "recall_at": {
            k: float(np.mean([recall_at_k(gold, pages, k) for gold, pages in rows]))
            for k in (5, 20, 50)
        },
        "mrr": float(np.mean([mrr(gold, pages) for gold, pages in rows])),
        "ndcg5": float(np.mean([ndcg_at_k(gold, pages, 5) for gold, pages in rows])),
        "n": len(relevant),
    }


def evaluate_coverage(
    questions: Sequence[CoverageQuestion],
    bm25_pages: Callable[[str], list[str]],
    channels: Mapping[str, CandidateChannel],
    *,
    limit: int = 50,
) -> dict[str, object]:
    """BM25 ve ek kanalları ilk-görülme sırasıyla birleştirip kapsama ölçer."""
    rows: list[tuple[set[str], list[str]]] = []
    by_slice: dict[str, list[tuple[set[str], list[str]]]] = defaultdict(list)
    diagnostics: list[dict[str, object]] = []
    for question in questions:
        if not question.answerable:
            continue
        source_pages: dict[str, list[str]] = {"bm25": bm25_pages(question.question)[:limit]}
        source_pages.update(
            {
                name: channel.candidate_pages(question.question, limit)[:limit]
                for name, channel in channels.items()
            }
        )
        pool = build_candidate_pool(
            source_pages["bm25"],
            [pages for name, pages in source_pages.items() if name != "bm25"],
            limit=limit,
        )
        gold = set(question.gold_page_ids)
        row = (gold, pool)
        rows.append(row)
        by_slice[question.slice].append(row)
        gold_sources = {
            page_id: [name for name, pages in source_pages.items() if page_id in pages]
            for page_id in question.gold_page_ids
            if page_id in pool
        }
        diagnostics.append(
            {
                "question_id": question.question_id,
                "slice": question.slice,
                "gold_page_ids": question.gold_page_ids,
                "candidate_pool": pool,
                "gold_sources": gold_sources,
            }
        )
    return {
        "overall": _summary(rows),
        "per_slice": {name: _summary(values) for name, values in sorted(by_slice.items())},
        "diagnostics": diagnostics,
    }


def select_dense_arm(arms: Mapping[str, Mapping[str, Any]]) -> str:
    """Önceden ilan edilmiş development tie-break; sonuç sonrası kural yok."""
    if not arms:
        raise ValueError("dense kol seçimi için başarılı kol yok")

    def key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, int, float, str]:
        name, arm = item
        paraphrase = float(arm.get("per_slice", {}).get("paraphrase", {}).get("coverage", 0.0))
        overall = float(arm["overall"]["coverage"])
        disk = int(arm.get("disk_bytes", 0))
        latency = float(arm.get("latency_ms", {}).get("p50", 0.0))
        return (-paraphrase, -overall, disk, latency, name)

    return min(arms.items(), key=key)[0]
