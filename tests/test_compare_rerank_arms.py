import pytest
from scripts.compare_rerank_arms import compare


def report(*, recall5: float, ranks: dict[str, int | None]) -> dict:
    diagnostics = [
        {
            "question_id": question_id,
            "slice": "paraphrase",
            "gold_rank": {"pinned": rank},
            "would_abstain": False,
        }
        for question_id, rank in ranks.items()
    ]
    return {
        "pinned": {
            "overall": {
                "n": len(ranks),
                "recall_at": {"5": recall5, "20": 1.0, "50": 1.0},
                "ndcg5": recall5,
                "ci_recall5": [0.0, 1.0],
            }
        },
        "unpinned": {"diagnostics": diagnostics},
        "latency_ms": {"rerank_p50": 1.0},
    }


def test_compare_uses_binary_top5_question_delta_instead_of_fractional_recall(capsys):
    base = report(recall5=0.5, ranks={"q1": None, "q2": 1})
    arm = report(recall5=0.75, ranks={"q1": 1, "q2": 1})

    verdict = compare(base, arm)

    output = capsys.readouterr().out
    assert "ilk-5 soru  1 -> 2  (+1, n=2)" in output
    assert "HÜKÜM: KEPT (tek soru" in output
    assert verdict.startswith("KEPT (tek soru")


def test_compare_rejects_mismatched_question_sets():
    base = report(recall5=0.5, ranks={"q1": None, "q2": 1})
    arm = report(recall5=0.5, ranks={"q1": 1, "q3": None})

    with pytest.raises(ValueError, match="question_id"):
        compare(base, arm)
