"""Sabit kaynak indeksinden Colab veya yerelde doğrulanabilir dense artefakt üretir."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.bench.dense_artifacts import (  # noqa: E402
    dense_model_key,
    page_ids_sha256,
    resume_dense_embeddings,
    sha256_file,
    write_dense_manifest,
)
from belge_gozu.retrieval.dense import (  # noqa: E402
    DENSE_MODELS,
    DenseModelOutOfMemory,
    DenseModelSpec,
    TransformerDenseEncoder,
    release_transformer_memory,
)
from belge_gozu.retrieval.hybrid import load_page_texts  # noqa: E402


def _full_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise ValueError("üretici Git commit'i 40 karakterli SHA olmalı")
    return value


def _resume_identity(spec: DenseModelSpec, page_ids: list[str]) -> dict[str, str]:
    return {
        "repo": spec.repo,
        "revision": spec.revision,
        "page_ids_sha256": page_ids_sha256(page_ids),
    }


def build_model_artifact(
    spec: DenseModelSpec,
    page_texts: Mapping[str, str],
    artifact_root: Path,
    *,
    page_texts_sha256: str,
    source_repo: str,
    source_revision: str,
    producer_git_commit: str,
    encoder: Any,
    max_batches: int | None = None,
) -> Path | None:
    """Tek modelin tamamlanan matrisini manifestler; kesintide `None` döner."""
    page_ids = list(page_texts)
    artifact_dir = Path(artifact_root) / dense_model_key(spec)
    embeddings = resume_dense_embeddings(
        encoder,
        [page_texts[page_id] for page_id in page_ids],
        artifact_dir,
        _resume_identity(spec, page_ids),
        batch_size=encoder.batch_size,
        max_batches=max_batches,
    )
    if embeddings is None:
        return None
    del embeddings
    write_dense_manifest(
        artifact_dir,
        spec=spec,
        page_ids=page_ids,
        page_texts_sha256=page_texts_sha256,
        source_repo=source_repo,
        source_revision=source_revision,
        producer_git_commit=producer_git_commit,
    )
    return artifact_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model", choices=sorted(DENSE_MODELS), required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("data/bench/dense-artifacts"))
    parser.add_argument("--device")
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches en az 1 olmalı")
    page_texts_path = args.index_dir / "page_texts.parquet"
    page_texts = load_page_texts(args.index_dir)
    spec = DENSE_MODELS[args.model]
    encoder = TransformerDenseEncoder(spec, device=args.device)
    try:
        encoder.preflight()
        artifact = build_model_artifact(
            spec,
            page_texts,
            args.artifact_root,
            page_texts_sha256=sha256_file(page_texts_path),
            source_repo=args.source_repo,
            source_revision=args.source_revision,
            producer_git_commit=_full_git_commit(),
            encoder=encoder,
            max_batches=args.max_batches,
        )
    except DenseModelOutOfMemory:
        print(json.dumps({"status": "skipped_oom", "model": args.model}, ensure_ascii=False))
        return 2
    finally:
        del encoder
        import torch

        release_transformer_memory(torch)
    if artifact is None:
        print(json.dumps({"status": "in_progress", "model": args.model}, ensure_ascii=False))
        return 0
    print(json.dumps({"status": "ok", "artifact": str(artifact)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
