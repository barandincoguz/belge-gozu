# ruff: noqa: E402

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_semantic_coverage_html import render_report  # pyright: ignore[reportMissingImports]


def test_renderer_includes_banner_metrics_and_escaped_question() -> None:
    report = {
        "mode": "development",
        "selected_dense_arm": "qwen3-embedding-8b",
        "arms": {
            "baseline": {
                "status": "ok",
                "overall": {"coverage": 0.8, "recall_at": {50: 0.8}, "n": 1},
                "per_slice": {"paraphrase": {"coverage": 0.5}},
                "diagnostics": [{"question_id": "q1", "candidate_pool": ["p1"]}],
            },
            "dense:qwen3-embedding-8b": {
                "status": "ok",
                "overall": {"coverage": 0.9},
                "per_slice": {"paraphrase": {"coverage": 0.9}},
                "diagnostics": [
                    {"question_id": "q1", "candidate_pool": ["<script>bad()</script>"]}
                ],
            },
        },
    }

    html = render_report(report)

    assert "DEVELOPMENT ONLY" in html
    assert "Paraphrase coverage" in html
    assert "qwen3-embedding-8b" in html
    assert "&lt;script&gt;bad()&lt;/script&gt;" in html
    assert "<script>bad()</script>" not in html
