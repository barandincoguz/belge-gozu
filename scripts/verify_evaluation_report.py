"""Verify persisted retrieval or answer-evaluation report integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.answer.calibrate import recipe_fingerprint  # noqa: E402
from belge_gozu.bench.report_validation import (  # noqa: E402
    validate_answer_report_payload,
    validate_provenance_hashes,
    validate_reranker_report_payload,
    validate_retrieval_report_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--kind", choices=("auto", "answer", "retrieval", "reranker"), default="auto"
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    try:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("rapor kökü JSON nesnesi olmalı")
        kind = args.kind
        if kind == "auto":
            if "metrics" in payload and "records" in payload:
                kind = "answer"
            elif "per_question" in payload and "pinned" in payload:
                kind = "reranker"
            else:
                kind = "retrieval"
        validate_provenance_hashes(payload, root=args.root)
        if kind == "answer":
            validate_answer_report_payload(
                payload,
                expected_recipe_fingerprint=recipe_fingerprint(),
                require_identity=True,
            )
        elif kind == "retrieval":
            validate_retrieval_report_payload(payload, require_bench=True)
        else:
            validate_reranker_report_payload(payload, require_benchmark=True)
    except Exception as exc:
        print(f"HATA: {args.report} doğrulanamadı: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {args.report} ({kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
