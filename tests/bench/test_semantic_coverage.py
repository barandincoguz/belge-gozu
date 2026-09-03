from dataclasses import dataclass

from belge_gozu.bench.semantic_coverage import evaluate_coverage, select_dense_arm


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    gold_page_ids: list[str]
    slice: str = "paraphrase"
    answerable: bool = True


class FixedChannel:
    def __init__(self, pages: list[str]) -> None:
        self.pages = pages

    def candidate_pages(self, query: str, limit: int) -> list[str]:
        assert query == "soru"
        return self.pages[:limit]


def test_evaluator_records_full_pool_coverage_and_gold_source() -> None:
    report = evaluate_coverage(
        questions=[Question("q1", "soru", ["gold"])],
        bm25_pages=lambda _: ["b1"],
        channels={"dense": FixedChannel(["gold"]), "mogan": FixedChannel(["b1"])},
    )

    assert report["overall"]["coverage"] == 1.0
    assert report["diagnostics"][0]["candidate_pool"] == ["b1", "gold"]
    assert report["diagnostics"][0]["gold_sources"] == {"gold": ["dense"]}


def test_dense_selection_prioritizes_paraphrase_then_overall_then_cost() -> None:
    arms = {
        "8b": {
            "per_slice": {"paraphrase": {"coverage": 1.0}},
            "overall": {"coverage": 1.0},
            "disk_bytes": 9,
            "latency_ms": {"p50": 9},
        },
        "4b": {
            "per_slice": {"paraphrase": {"coverage": 1.0}},
            "overall": {"coverage": 1.0},
            "disk_bytes": 8,
            "latency_ms": {"p50": 1},
        },
    }

    assert select_dense_arm(arms) == "4b"
