"""Qwen3 dense ve genişletme kollarını yalnız development bench'te ölçer."""

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
from typing import Protocol

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.bench.dataset import load_bench  # noqa: E402
from belge_gozu.bench.semantic_coverage import evaluate_coverage, select_dense_arm  # noqa: E402
from belge_gozu.config import Settings  # noqa: E402
from belge_gozu.index.manifest import index_revision, read_manifest  # noqa: E402
from belge_gozu.provenance import git_commit  # noqa: E402
from belge_gozu.retrieval.dense import (  # noqa: E402
    DENSE_MODELS,
    DenseModelOutOfMemory,
    DensePageIndex,
    TransformerDenseEncoder,
    release_transformer_memory,
)
from belge_gozu.retrieval.expand import (  # noqa: E402
    EXPANDER_REPO,
    EXPANDER_REVISION,
    ExpansionModelOutOfMemory,
    ExpansionRecord,
    LocalQueryExpander,
    load_expansion_cache,
    prompt_fingerprint,
    question_fingerprint,
    write_expansion_cache,
)
from belge_gozu.retrieval.hybrid import load_page_texts, load_text_channel  # noqa: E402
from belge_gozu.retrieval.late import load_late_channel  # noqa: E402
from belge_gozu.retrieval.text import rank_order, route_window, routed_docs  # noqa: E402

CANDIDATE_LIMIT = 50


class Question(Protocol):
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


class TextChannel(Protocol):
    page_ids: list[str]

    def scores(self, query: str) -> np.ndarray: ...


class LateChannel(Protocol):
    def candidate_pages(self, query: str, limit: int) -> list[str]: ...


class _CachedChannel:
    def __init__(self, pages: Mapping[str, list[str]]) -> None:
        self.pages = pages

    def candidate_pages(self, query: str, limit: int) -> list[str]:
        return self.pages[query][:limit]


def evaluate_cached_sources(
    questions: Sequence[Question], sources: Mapping[str, Mapping[str, list[str]]]
) -> dict[str, object]:
    """Önceden hesaplanan listelerle saf, ilk-görülme kapsama değerlendirmesi."""
    if "bm25" not in sources:
        raise ValueError("semantic kaynaklarda bm25 zorunlu")
    return evaluate_coverage(
        questions,
        lambda query: sources["bm25"][query],
        {name: _CachedChannel(pages) for name, pages in sources.items() if name != "bm25"},
        limit=CANDIDATE_LIMIT,
    )


def _chunk_pages(index_dir: Path, page_ids: list[str]) -> dict[str, tuple[str, ...]]:
    chunks = pd.read_parquet(index_dir / "chunks.parquet")
    if "chunk_id" not in chunks or "page_ids" not in chunks:
        raise ValueError("chunks.parquet chunk_id ve page_ids sütunlarını içermeli")
    if chunks["chunk_id"].duplicated().any():
        raise ValueError("chunks.parquet yinelenen chunk_id içeriyor")
    result = {
        str(chunk_id): tuple(str(page_id) for page_id in page_list)
        for chunk_id, page_list in zip(chunks["chunk_id"], chunks["page_ids"], strict=True)
    }
    unknown = sorted({page_id for pages in result.values() for page_id in pages} - set(page_ids))
    if unknown:
        raise ValueError(f"chunks.parquet bilinmeyen sayfa taşıyor: {unknown[:3]}")
    return result


def _bm25_pages(
    text: TextChannel, doc_names: Mapping[str, frozenset[str]], query: str
) -> list[str]:
    scores = np.asarray(text.scores(query), dtype=np.float64)
    if scores.shape != (len(text.page_ids),):
        raise ValueError("BM25 skorları page_ids ile hizalı olmalı")
    order = [text.page_ids[int(index)] for index in rank_order(scores)]
    return route_window(order, routed_docs(query, doc_names))


def _answerable(questions: Sequence[Question]) -> list[Question]:
    result = [question for question in questions if question.answerable]
    if not result:
        raise ValueError("semantic coverage için cevaplanabilir soru yok")
    return result


def _base_sources(
    questions: Sequence[Question],
    text: TextChannel,
    doc_names: Mapping[str, frozenset[str]],
    late_channels: Mapping[str, LateChannel],
) -> dict[str, dict[str, list[str]]]:
    sources = {"bm25": {}, **{name: {} for name in late_channels}}
    for question in questions:
        sources["bm25"][question.question] = _bm25_pages(text, doc_names, question.question)
        for name, channel in late_channels.items():
            sources[name][question.question] = channel.candidate_pages(
                question.question, CANDIDATE_LIMIT
            )
    return sources


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def _write_embeddings(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        np.save(handle, embeddings)
        temp_name = handle.name
    os.replace(temp_name, path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dense_arm(
    *,
    name: str,
    page_ids: list[str],
    page_texts: Mapping[str, str],
    questions: Sequence[Question],
    base_sources: Mapping[str, Mapping[str, list[str]]],
    artifact_dir: Path,
    device: str | None,
) -> tuple[dict[str, object], dict[str, list[str]] | None, Path | None]:
    spec = DENSE_MODELS[name]
    encoder = TransformerDenseEncoder(spec, device=device)
    try:
        encoder.preflight()
        started = time.perf_counter()
        embeddings = encoder.encode_passages([page_texts[page_id] for page_id in page_ids])
        elapsed_ms = (time.perf_counter() - started) * 1000
        index = DensePageIndex(page_ids, embeddings)
        candidates = {
            question.question: index.candidate_pages(
                encoder.encode_queries([question.question])[0], CANDIDATE_LIMIT
            )
            for question in questions
        }
        arm_dir = artifact_dir / name
        vector_path = arm_dir / "embeddings.npy"
        _write_embeddings(vector_path, embeddings)
        _write_json(
            arm_dir / "dense.json",
            {
                "model_repo": spec.repo,
                "model_revision": spec.revision,
                "instruction": spec.instruction,
                "page_ids_sha256": hashlib.sha256("\n".join(page_ids).encode()).hexdigest(),
                "dimension": int(embeddings.shape[1]),
            },
        )
        report = evaluate_cached_sources(questions, {**base_sources, "dense": candidates})
        report.update(
            {
                "status": "ok",
                "model": {"repo": spec.repo, "revision": spec.revision},
                "disk_bytes": vector_path.stat().st_size,
                "latency_ms": {
                    "p50": elapsed_ms / len(page_ids),
                    "p95": elapsed_ms / len(page_ids),
                },
                "artifact": {"path": str(vector_path), "sha256": _sha256_file(vector_path)},
            }
        )
        return report, candidates, vector_path
    except DenseModelOutOfMemory as exc:
        return {"status": "skipped_oom", "reason": str(exc)}, None, None
    finally:
        del encoder
        import torch

        release_transformer_memory(torch)


def _expanded_queries(
    questions: Sequence[Question], cache_path: Path, device: str | None
) -> dict[str, str]:
    cached = load_expansion_cache(cache_path)
    records = dict(cached)
    expander = LocalQueryExpander(device=device)
    try:
        expander.preflight()
        for question in questions:
            existing = records.get(question.question_id)
            question_sha = question_fingerprint(question.question)
            if existing is not None:
                if existing.question_sha256 != question_sha:
                    raise ValueError(
                        f"genişletme cache soru hash'i uyuşmuyor: {question.question_id}"
                    )
                continue
            records[question.question_id] = ExpansionRecord(
                question_id=question.question_id,
                question_sha256=question_sha,
                prompt_fingerprint=prompt_fingerprint(),
                model_revision=EXPANDER_REVISION,
                expansion=expander.expand(question.question),
            )
        write_expansion_cache(cache_path, list(records.values()))
        return {
            question.question: records[question.question_id].expansion
            for question in questions
        }
    finally:
        del expander
        import torch

        release_transformer_memory(torch)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--min-verification", default="human")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("data/bench/artifacts/semantic-v1"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = Settings()
    device = None if settings.device == "auto" else settings.device
    questions = _answerable(
        load_bench(args.bench, only_verified=True, min_verification=args.min_verification)
    )
    index_dir = settings.index_dir
    page_texts = load_page_texts(index_dir)
    page_ids = list(page_texts)
    text, doc_names = load_text_channel(index_dir, page_ids)
    chunk_pages = _chunk_pages(index_dir, page_ids)

    late = {
        "mogan": load_late_channel(settings.late_mogan_index_dir, chunk_pages, device=device),
        "colmm": load_late_channel(settings.late_colmm_index_dir, chunk_pages, device=device),
    }
    base_sources = _base_sources(questions, text, doc_names, late)
    baseline = evaluate_cached_sources(questions, base_sources)
    del late
    import torch

    release_transformer_memory(torch)

    arms: dict[str, dict[str, object]] = {"baseline": {"status": "ok", **baseline}}
    dense_candidates: dict[str, dict[str, list[str]]] = {}
    dense_artifacts: dict[str, Path] = {}
    for name in DENSE_MODELS:
        arm, candidates, artifact = _dense_arm(
            name=name,
            page_ids=page_ids,
            page_texts=page_texts,
            questions=questions,
            base_sources=base_sources,
            artifact_dir=args.artifacts,
            device=device,
        )
        arms[f"dense:{name}"] = arm
        if candidates is not None and artifact is not None:
            dense_candidates[name] = candidates
            dense_artifacts[name] = artifact

    winner = (
        select_dense_arm({name: arms[f"dense:{name}"] for name in dense_candidates})
        if dense_candidates
        else None
    )
    expansion_status: dict[str, object] = {"status": "skipped_no_dense"}
    if winner is not None:
        try:
            expansions = _expanded_queries(questions, args.cache, device)
            late = {
                "mogan-expanded": load_late_channel(
                    settings.late_mogan_index_dir, chunk_pages, device=device
                ),
                "colmm-expanded": load_late_channel(
                    settings.late_colmm_index_dir, chunk_pages, device=device
                ),
            }
            expanded_base = _base_sources(
                questions,
                text,
                doc_names,
                {name: channel for name, channel in late.items()},
            )
            for question in questions:
                expanded_query = expansions[question.question]
                expanded_base["bm25"][question.question] = _bm25_pages(
                    text, doc_names, expanded_query
                )
                for name, channel in late.items():
                    expanded_base[name][question.question] = channel.candidate_pages(
                        expanded_query, CANDIDATE_LIMIT
                    )
            del late
            release_transformer_memory(torch)

            spec = DENSE_MODELS[winner]
            encoder = TransformerDenseEncoder(spec, device=device)
            try:
                encoder.preflight()
                index = DensePageIndex(page_ids, np.load(dense_artifacts[winner]))
                expanded_dense = {
                    question.question: index.candidate_pages(
                        encoder.encode_queries([expansions[question.question]])[0], CANDIDATE_LIMIT
                    )
                    for question in questions
                }
            finally:
                del encoder
                release_transformer_memory(torch)
            expanded_sources = {
                **base_sources,
                "dense": dense_candidates[winner],
                "bm25-expanded": expanded_base["bm25"],
                "mogan-expanded": expanded_base["mogan-expanded"],
                "colmm-expanded": expanded_base["colmm-expanded"],
                "dense-expanded": expanded_dense,
            }
            expansion_status = {
                "status": "ok",
                **evaluate_cached_sources(questions, expanded_sources),
                "model": {"repo": EXPANDER_REPO, "revision": EXPANDER_REVISION},
                "selected_dense": winner,
                "cache": {"path": str(args.cache), "sha256": _sha256_file(args.cache)},
            }
        except (DenseModelOutOfMemory, ExpansionModelOutOfMemory) as exc:
            expansion_status = {
                "status": "skipped_oom",
                "reason": str(exc),
                "selected_dense": winner,
            }
    arms[f"dense:{winner}+expand" if winner else "expand"] = expansion_status

    manifest = read_manifest(index_dir)
    report: dict[str, object] = {
        "schema_version": 1,
        "mode": "development",
        "git_commit": git_commit(),
        "candidate_limit": CANDIDATE_LIMIT,
        "arms": arms,
        "selected_dense": winner,
        "selection": {
            "min_verification": args.min_verification,
            "answerable_questions": len(questions),
        },
        "provenance": {
            "benchmark": {"path": str(args.bench), "sha256": _sha256_file(args.bench)},
            "index": {
                "path": str(index_dir),
                "revision": index_revision(manifest) if manifest else None,
                "page_texts_sha256": _sha256_file(index_dir / "page_texts.parquet"),
            },
        },
    }
    _write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
