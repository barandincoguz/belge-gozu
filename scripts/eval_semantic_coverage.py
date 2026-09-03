"""Qwen3 dense ve sorgu genişletme kollarını yalnız developmentta ölçer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.bench.dataset import BenchQuestion, load_bench  # noqa: E402
from belge_gozu.bench.semantic_coverage import evaluate_coverage, select_dense_arm  # noqa: E402
from belge_gozu.config import Settings  # noqa: E402
from belge_gozu.index.manifest import index_revision, read_manifest  # noqa: E402
from belge_gozu.provenance import git_commit  # noqa: E402
from belge_gozu.retrieval.dense import (  # noqa: E402
    DENSE_MODELS,
    DenseModelOutOfMemory,
    DenseModelSpec,
    DensePageIndex,
    TransformerDenseEncoder,
    release_transformer_memory,
)
from belge_gozu.retrieval.expand import (  # noqa: E402
    ExpansionModelOutOfMemory,
    ExpansionRecord,
    LocalQueryExpander,
    load_expansion_cache,
    question_fingerprint,
    write_expansion_cache,
)
from belge_gozu.retrieval.hybrid import load_page_texts, load_text_channel  # noqa: E402
from belge_gozu.retrieval.late import load_late_channel  # noqa: E402
from belge_gozu.retrieval.text import rank_order, route_window, routed_docs  # noqa: E402


class _CachedChannel:
    def __init__(self, pages_by_query: Mapping[str, list[str]]) -> None:
        self._pages_by_query = pages_by_query

    def candidate_pages(self, query: str, limit: int) -> list[str]:
        return self._pages_by_query[query][:limit]


def evaluate_cached_sources(
    questions: Sequence[BenchQuestion], source_pages: Mapping[str, Mapping[str, list[str]]]
) -> dict[str, object]:
    """Önceden hesaplanan kanalları ilk-görülme sırasıyla ölçer."""
    if "bm25" not in source_pages:
        raise ValueError("semantic kaynaklarında bm25 zorunlu")
    return evaluate_coverage(
        questions,
        lambda query: source_pages["bm25"][query],
        {
            name: _CachedChannel(pages_by_query)
            for name, pages_by_query in source_pages.items()
            if name != "bm25"
        },
    )


def _chunk_pages(index_dir: Path, page_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
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


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        np.save(handle, value)
        temporary = handle.name
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bm25_pages(text: Any, doc_names: Mapping[str, frozenset[str]], query: str) -> list[str]:
    scores = np.asarray(text.scores(query), dtype=np.float64)
    if scores.shape != (len(text.page_ids),):
        raise ValueError("BM25 skorları page_ids ile hizalı olmalı")
    order = [text.page_ids[int(index)] for index in rank_order(scores)]
    return route_window(order, routed_docs(query, doc_names))


def _baseline_sources(
    questions: Sequence[BenchQuestion],
    text: Any,
    doc_names: Mapping[str, frozenset[str]],
    late_channels: Mapping[str, Any],
    query_for_question: Mapping[str, str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    sources = {"bm25": {}, **{name: {} for name in late_channels}}
    for question in questions:
        if not question.answerable:
            continue
        search_query = (query_for_question or {}).get(question.question, question.question)
        sources["bm25"][question.question] = _bm25_pages(text, doc_names, search_query)
        for name, channel in late_channels.items():
            sources[name][question.question] = channel.candidate_pages(search_query, 50)
    return sources


def _dense_arm(
    spec: DenseModelSpec,
    *,
    questions: Sequence[BenchQuestion],
    page_ids: list[str],
    page_texts: Mapping[str, str],
    baseline: Mapping[str, Mapping[str, list[str]]],
    artifact_root: Path,
    device: str | None,
) -> tuple[dict[str, object], dict[str, list[str]] | None]:
    started = time.perf_counter()
    encoder = TransformerDenseEncoder(spec, device=device)
    try:
        encoder.preflight()
        embeddings = encoder.encode_passages([page_texts[page_id] for page_id in page_ids])
        index = DensePageIndex(page_ids, embeddings)
        dense_pages = {
            question.question: index.candidate_pages(encoder.encode_queries([question.question])[0])
            for question in questions
            if question.answerable
        }
    except DenseModelOutOfMemory:
        return {
            "status": "skipped_oom",
            "model": {"repo": spec.repo, "revision": spec.revision},
        }, None
    finally:
        del encoder
        import torch

        release_transformer_memory(torch)

    model_dir = artifact_root / spec.repo.rsplit("/", 1)[-1]
    embeddings_path = model_dir / "embeddings.npy"
    _atomic_npy(embeddings_path, embeddings)
    _atomic_json(
        model_dir / "dense.json",
        {
            "repo": spec.repo,
            "revision": spec.revision,
            "page_ids_sha256": hashlib.sha256("\n".join(page_ids).encode()).hexdigest(),
            "dimension": int(embeddings.shape[1]),
        },
    )
    report = evaluate_cached_sources(questions, {**baseline, "dense": dense_pages})
    report.update(
        {
            "status": "ok",
            "model": {"repo": spec.repo, "revision": spec.revision},
            "disk_bytes": embeddings_path.stat().st_size,
            "latency_ms": {"p50": (time.perf_counter() - started) * 1000 / len(dense_pages)},
            "artifact": {"path": str(model_dir), "sha256": _sha256_file(embeddings_path)},
        }
    )
    return report, dense_pages


def _expansions(
    questions: Sequence[BenchQuestion], cache_path: Path, device: str | None
) -> dict[str, str]:
    records = load_expansion_cache(cache_path)
    pending = [
        question
        for question in questions
        if question.answerable and question.question_id not in records
    ]
    if pending:
        expander = LocalQueryExpander(device=device)
        try:
            expander.preflight()
            for question in pending:
                records[question.question_id] = ExpansionRecord(
                    question_id=question.question_id,
                    question_sha256=question_fingerprint(question.question),
                    prompt_fingerprint=expander_prompt_fingerprint(),
                    model_revision=expander_revision(),
                    expansion=expander.expand(question.question),
                )
        finally:
            del expander
            import torch

            release_transformer_memory(torch)
        write_expansion_cache(cache_path, list(records.values()))
    result: dict[str, str] = {}
    for question in questions:
        if not question.answerable:
            continue
        record = records.get(question.question_id)
        if record is None or record.question_sha256 != question_fingerprint(question.question):
            raise ValueError(f"genişletme önbellek soru hash'i uyuşmuyor: {question.question_id}")
        result[question.question] = record.expansion
    return result


def expander_prompt_fingerprint() -> str:
    from belge_gozu.retrieval.expand import prompt_fingerprint

    return prompt_fingerprint()


def expander_revision() -> str:
    from belge_gozu.retrieval.expand import EXPANDER_REVISION

    return EXPANDER_REVISION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--min-verification", default="human")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dense-artifacts", type=Path, default=Path("data/bench/dense-artifacts"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = Settings()
    device = None if settings.device == "auto" else settings.device
    questions = load_bench(args.bench, only_verified=True, min_verification=args.min_verification)
    index_dir = settings.index_dir
    page_texts = load_page_texts(index_dir)
    page_ids = list(page_texts)
    text, doc_names = load_text_channel(index_dir, page_ids)
    chunk_pages = _chunk_pages(index_dir, page_ids)
    late_channels = {
        "mogan": load_late_channel(settings.late_mogan_index_dir, chunk_pages, device=device),
        "colmm": load_late_channel(settings.late_colmm_index_dir, chunk_pages, device=device),
    }
    baseline = _baseline_sources(questions, text, doc_names, late_channels)
    baseline_report = evaluate_cached_sources(questions, baseline)
    del late_channels
    import torch

    release_transformer_memory(torch)

    arms: dict[str, dict[str, object]] = {"baseline": {"status": "ok", **baseline_report}}
    dense_pages_by_arm: dict[str, dict[str, list[str]]] = {}
    for name, spec in DENSE_MODELS.items():
        arm, dense_pages = _dense_arm(
            spec,
            questions=questions,
            page_ids=page_ids,
            page_texts=page_texts,
            baseline=baseline,
            artifact_root=args.dense_artifacts,
            device=device,
        )
        arms[f"dense:{name}"] = arm
        if dense_pages is not None:
            dense_pages_by_arm[name] = dense_pages

    winner = (
        select_dense_arm({name: arms[f"dense:{name}"] for name in dense_pages_by_arm})
        if dense_pages_by_arm
        else None
    )
    if winner is None:
        arms["expand"] = {"status": "skipped_no_dense"}
    else:
        try:
            expanded_queries = _expansions(questions, args.cache, device)
        except ExpansionModelOutOfMemory:
            arms[f"dense:{winner}+expand"] = {"status": "skipped_oom"}
        else:
            late_channels = {
                "mogan": load_late_channel(
                    settings.late_mogan_index_dir, chunk_pages, device=device
                ),
                "colmm": load_late_channel(
                    settings.late_colmm_index_dir, chunk_pages, device=device
                ),
            }
            expanded = _baseline_sources(
                questions, text, doc_names, late_channels, query_for_question=expanded_queries
            )
            expanded_baseline = {
                "bm25-expansion": expanded["bm25"],
                "mogan-expansion": expanded["mogan"],
                "colmm-expansion": expanded["colmm"],
            }
            del late_channels
            release_transformer_memory(torch)
            spec = DENSE_MODELS[winner]
            encoder = TransformerDenseEncoder(spec, device=device)
            try:
                encoder.preflight()
                embeddings = np.load(
                    args.dense_artifacts / spec.repo.rsplit("/", 1)[-1] / "embeddings.npy"
                )
                index = DensePageIndex(page_ids, embeddings)
                expanded_dense = {
                    question.question: index.candidate_pages(
                        encoder.encode_queries([expanded_queries[question.question]])[0]
                    )
                    for question in questions
                    if question.answerable
                }
            except DenseModelOutOfMemory:
                arms[f"dense:{winner}+expand"] = {"status": "skipped_oom"}
            else:
                expansion_sources = {
                    "bm25": baseline["bm25"],
                    "mogan": baseline["mogan"],
                    "colmm": baseline["colmm"],
                    "dense": dense_pages_by_arm[winner],
                    "bm25-expansion": expanded_baseline["bm25-expansion"],
                    "mogan-expansion": expanded_baseline["mogan-expansion"],
                    "colmm-expansion": expanded_baseline["colmm-expansion"],
                    "dense-expansion": expanded_dense,
                }
                arms[f"dense:{winner}+expand"] = {
                    "status": "ok",
                    "model": {"repo": spec.repo, "revision": spec.revision},
                    **evaluate_cached_sources(questions, expansion_sources),
                }
            finally:
                del encoder
                release_transformer_memory(torch)

    manifest = read_manifest(index_dir)
    report: dict[str, object] = {
        "schema_version": 1,
        "mode": "development",
        "git_commit": git_commit(),
        "selected_dense_arm": winner,
        "arms": arms,
        "selection": {"only_verified": True, "min_verification": args.min_verification},
        "benchmark": {"path": str(args.bench), "sha256": _sha256_file(args.bench)},
        "index": {
            "path": str(index_dir),
            "revision": index_revision(manifest) if manifest else None,
            "page_texts_sha256": _sha256_file(index_dir / "page_texts.parquet"),
        },
    }
    _atomic_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
