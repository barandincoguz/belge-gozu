"""Sabit Hugging Face commit'inden doğrulanmış dense artefakt indirir."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.bench.dense_artifact_hub import pull_dense_artifact  # noqa: E402
from belge_gozu.bench.dense_artifacts import DenseArtifactExpectation, sha256_file  # noqa: E402
from belge_gozu.retrieval.dense import DENSE_MODELS  # noqa: E402
from belge_gozu.retrieval.hybrid import load_page_texts  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="ayrı dense artefakt Dataset deposu")
    parser.add_argument("--revision", required=True, help="40 karakterli Hub commit SHA")
    parser.add_argument("--model", choices=sorted(DENSE_MODELS), required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("data/bench/dense-artifacts"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    page_texts = load_page_texts(args.index_dir)
    sha = pull_dense_artifact(
        repo_id=args.repo,
        revision=args.revision,
        model_key=args.model,
        destination_root=args.artifact_root,
        expectation=DenseArtifactExpectation(
            model=DENSE_MODELS[args.model],
            page_ids=list(page_texts),
            page_texts_sha256=sha256_file(args.index_dir / "page_texts.parquet"),
        ),
        token=os.environ.get("HF_TOKEN", ""),
    )
    print(f"dense artefakt indirildi; commit={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
