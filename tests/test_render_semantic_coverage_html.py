# ruff: noqa: E402, E501

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_semantic_coverage_html import render_report  # pyright: ignore[reportMissingImports]


def test_renderer_marks_development_and_escapes_diagnostics() -> None:
    report = {
        "mode": "development",
        "selected_dense": "qwen3-embedding-8b",
        "arms": {
            "baseline": {"status": "ok", "overall": {"coverage": 0.8}, "per_slice": {"paraphrase": {"coverage": 0.5}}},
            "dense:qwen3-embedding-8b": {"status": "ok", "overall": {"coverage": 0.9}, "per_slice": {"paraphrase": {"coverage": 0.9}}, "diagnostics": [{"question_id": "q1", "candidate_pool": ["<script>bad()</script>"]}]},
        },
    }

    html = render_report(report)

    assert "DEVELOPMENT ONLY" in html
    assert "Paraphrase R@50" in html
    assert "--value:90.00%" in html
    assert "&lt;script&gt;bad()&lt;/script&gt;" in html
    assert "<script>bad()</script>" not in html
