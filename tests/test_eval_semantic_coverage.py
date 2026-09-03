# ruff: noqa: E402

import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from eval_semantic_coverage import evaluate_cached_sources  # pyright: ignore[reportMissingImports]


@dataclass(frozen=True)
class Question:
    question_id: str = "q1"
    question: str = "soru"
    answerable: bool = True
    gold_page_ids: list[str] | None = None
    slice: str = "paraphrase"

    def __post_init__(self) -> None:
        if self.gold_page_ids is None:
            object.__setattr__(self, "gold_page_ids", ["gold"])


def test_cached_sources_preserve_declared_first_seen_order() -> None:
    report = evaluate_cached_sources(
        [Question()],
        {
            "bm25": {"soru": ["bm25"]},
            "mogan": {"soru": ["gold"]},
            "dense": {"soru": ["dense-only"]},
        },
    )

    assert report["diagnostics"][0]["candidate_pool"] == ["bm25", "gold", "dense-only"]
    assert report["overall"]["coverage"] == 1.0
