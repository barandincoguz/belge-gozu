"""BM25-sabit ve serbest BGE rerank kollarını yalnız offline karşılaştırır."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.answer.calibrate import git_blob_sha, sha256_file  # noqa: E402
from belge_gozu.bench.dataset import load_bench  # noqa: E402
from belge_gozu.bench.metrics import bootstrap_ci, mrr, ndcg_at_k, recall_at_k  # noqa: E402
from belge_gozu.config import Settings  # noqa: E402
from belge_gozu.index.manifest import index_revision, read_manifest  # noqa: E402
from belge_gozu.provenance import git_commit  # noqa: E402
from belge_gozu.retrieval.candidates import build_candidate_pool  # noqa: E402
from belge_gozu.retrieval.hybrid import load_page_texts, load_text_channel  # noqa: E402
from belge_gozu.retrieval.late import load_late_channel  # noqa: E402
from belge_gozu.retrieval.rerank import (  # noqa: E402
    PageReranker,
    TransformerPageReranker,
    compare_rerankings,
)
from belge_gozu.retrieval.text import rank_order, route_window, routed_docs  # noqa: E402

KS = (5, 20, 50)
CANDIDATE_LIMIT = 50


class TextChannel(Protocol):
    page_ids: list[str]

    def scores(self, query: str) -> np.ndarray: ...


class LateCandidateChannel(Protocol):
    def candidate_pages(self, query: str, limit: int) -> list[str]: ...


class Question(Protocol):
    question_id: str
    question: str
    answerable: bool
    gold_page_ids: list[str]

    @property
    def slice(self) -> str: ...


class ArmRows(TypedDict):
    rankings: list[list[str]]
    relevant: list[set[str]]
    slices: list[str]


def _metrics(rows: ArmRows) -> dict[str, object]:
    ranked, relevant = rows["rankings"], rows["relevant"]
    recalls = {
        k: [recall_at_k(gold, pages, k) for gold, pages in zip(relevant, ranked, strict=True)]
        for k in KS
    }
    mrr_values = [mrr(gold, pages) for gold, pages in zip(relevant, ranked, strict=True)]
    ndcg5_values = [
        ndcg_at_k(gold, pages, 5) for gold, pages in zip(relevant, ranked, strict=True)
    ]
    return {
        "recall_at": {k: float(np.mean(values)) for k, values in recalls.items()},
        "mrr": float(np.mean(mrr_values)),
        "ndcg5": float(np.mean(ndcg5_values)),
        "n": len(ranked),
        "ci_recall5": list(bootstrap_ci(recalls[5])),
    }


def _metrics_by_slice(rows: ArmRows) -> dict[str, dict[str, object]]:
    grouped: dict[str, ArmRows] = {}
    for index, slice_name in enumerate(rows["slices"]):
        group = grouped.setdefault(slice_name, {"rankings": [], "relevant": [], "slices": []})
        group["rankings"].append(rows["rankings"][index])
        group["relevant"].append(rows["relevant"][index])
        group["slices"].append(slice_name)
    return {slice_name: _metrics(group) for slice_name, group in sorted(grouped.items())}


def _arm_report(rows: ArmRows) -> dict[str, object]:
    return {"overall": _metrics(rows), "per_slice": _metrics_by_slice(rows)}


def run_comparison(
    *,
    questions: Sequence[Question],
    text: TextChannel,
    doc_names: Mapping[str, frozenset[str]],
    page_texts: Mapping[str, str],
    late_channels: Sequence[LateCandidateChannel],
    reranker: PageReranker,
    threshold: float = 10.6,
    candidate_limit: int = CANDIDATE_LIMIT,
) -> dict[str, object]:
    """Tek skor füzyonu yapmadan P/U sıralamalarını ölçer; dosya/model yüklemez."""
    answerable = [question for question in questions if question.answerable]
    if not answerable:
        raise ValueError("rerank karşılaştırması için cevaplanabilir soru yok")

    candidate_rows: ArmRows = {"rankings": [], "relevant": [], "slices": []}
    pinned_rows: ArmRows = {"rankings": [], "relevant": [], "slices": []}
    unpinned_rows: ArmRows = {"rankings": [], "relevant": [], "slices": []}
    diagnostics: list[dict[str, object]] = []
    rerank_ms: list[float] = []

    for question in answerable:
        scores = np.asarray(text.scores(question.question), dtype=np.float64)
        if scores.shape != (len(text.page_ids),):
            raise ValueError("BM25 skorları page_ids ile hizalı olmalı")
        bm25_order = [text.page_ids[int(index)] for index in rank_order(scores)]
        bm25_routed = route_window(bm25_order, routed_docs(question.question, doc_names))
        bm25_scores = dict(zip(text.page_ids, scores.tolist(), strict=True))
        late_pages = [
            channel.candidate_pages(question.question, candidate_limit) for channel in late_channels
        ]
        pool = build_candidate_pool(bm25_routed, late_pages, limit=candidate_limit)

        started = time.perf_counter()
        comparison = compare_rerankings(
            question.question, pool, page_texts, bm25_scores, reranker, threshold=threshold
        )
        rerank_ms.append((time.perf_counter() - started) * 1000)
        relevant = set(question.gold_page_ids)
        for rows, ranking in (
            (candidate_rows, pool),
            (pinned_rows, list(comparison.pinned_pages)),
            (unpinned_rows, list(comparison.unpinned_pages)),
        ):
            rows["rankings"].append(ranking)
            rows["relevant"].append(relevant)
            rows["slices"].append(question.slice)
        diagnostics.append(
            {
                "question_id": question.question_id,
                "slice": question.slice,
                "candidate_pool_size": len(pool),
                "candidate_pool": pool,
                "pinned_top1": comparison.pinned_pages[0],
                "unpinned_top1": comparison.unpinned_pages[0],
                "bm25_top1_rank_unpinned": comparison.bm25_top1_rank_unpinned,
                "unpinned_top1_bm25_score": comparison.unpinned_top1_bm25_score,
                "would_abstain": comparison.would_abstain,
            }
        )

    report = {
        "candidate_limit": candidate_limit,
        "threshold": threshold,
        "candidate_pool": _arm_report(candidate_rows),
        "pinned": _arm_report(pinned_rows),
        "unpinned": {**_arm_report(unpinned_rows), "diagnostics": diagnostics},
        "latency_ms": {
            "rerank_p50": float(np.percentile(rerank_ms, 50)),
            "rerank_p95": float(np.percentile(rerank_ms, 95)),
        },
    }
    return report


def _chunk_pages(index_dir: Path, page_ids: list[str]) -> dict[str, tuple[str, ...]]:
    chunks = pd.read_parquet(index_dir / "chunks.parquet")
    if "chunk_id" not in chunks or "page_ids" not in chunks:
        raise ValueError("chunks.parquet chunk_id ve page_ids sütunlarını içermeli")
    if chunks["chunk_id"].duplicated().any():
        raise ValueError("chunks.parquet yinelenen chunk_id içeriyor")
    mapped = {
        str(chunk_id): tuple(str(page_id) for page_id in page_list)
        for chunk_id, page_list in zip(chunks["chunk_id"], chunks["page_ids"], strict=True)
    }
    unknown = sorted({page_id for pages in mapped.values() for page_id in pages} - set(page_ids))
    if unknown:
        raise ValueError(f"chunks.parquet bilinmeyen sayfa taşıyor: {unknown[:3]}")
    return mapped


def _write_atomic_json(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def _provenance(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path), "git_blob_sha": git_blob_sha(path)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--min-verification", default="human")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--yes-final-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.final and not args.yes_final_gate:
        raise SystemExit("--final yalnız --yes-final-gate ile çalıştırılabilir")

    settings = Settings()
    index_dir = settings.index_dir
    page_texts = load_page_texts(index_dir)
    page_ids = list(page_texts)
    text, doc_names = load_text_channel(index_dir, page_ids)
    chunk_pages = _chunk_pages(index_dir, page_ids)
    device = None if settings.device == "auto" else settings.device
    late_channels = (
        load_late_channel(settings.late_mogan_index_dir, chunk_pages, device=device),
        load_late_channel(settings.late_colmm_index_dir, chunk_pages, device=device),
    )
    reranker = TransformerPageReranker(device=device)
    questions = load_bench(args.bench, only_verified=True, min_verification=args.min_verification)
    answerable_questions = sum(question.answerable for question in questions)
    report = run_comparison(
        questions=questions,
        text=text,
        doc_names=doc_names,
        page_texts=page_texts,
        late_channels=late_channels,
        reranker=reranker,
    )
    manifest = read_manifest(index_dir)
    report.update(
        {
            "schema_version": 1,
            "mode": "final" if args.final else "development",
            "git_commit": git_commit(),
            "benchmark": _provenance(args.bench),
            "index": {
                "path": str(index_dir),
                "revision": index_revision(manifest) if manifest else None,
                "page_texts": _provenance(index_dir / "page_texts.parquet"),
                "chunks": _provenance(index_dir / "chunks.parquet"),
            },
            "model": {
                "repo": reranker.repo,
                "revision": reranker.revision,
                "device": reranker._device,
                "max_length": reranker.max_length,
                "batch_size": reranker.batch_size,
            },
            "selection": {
                "only_verified": True,
                "min_verification": args.min_verification,
                "answerable_questions": answerable_questions,
            },
        }
    )
    _write_atomic_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
