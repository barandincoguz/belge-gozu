from datetime import UTC, datetime
from typing import Literal, cast

import pytest

from belge_gozu.bench.answer_eval import (
    AnswerRecord,
    ClaimRecord,
    run_answer_eval,
)

ClaimVerdict = Literal["supported", "unsupported", "belirsiz"]
AnswerStatus = Literal["answered", "abstained", "degraded"]


def answer_record(
    *,
    question_id: str = "q1",
    answerable: bool,
    verdicts: list[tuple[ClaimVerdict, list[int]]],
    status: AnswerStatus = "answered",
    honest_miss: bool = False,
    n_claims: int | None = None,
) -> AnswerRecord:
    claims = tuple(
        ClaimRecord(
            claim_id=f"c{i}",
            verdict=verdict,
            gerekce="stub",
            cited_sources=tuple(sources),
        )
        for i, (verdict, sources) in enumerate(verdicts, start=1)
    )
    return AnswerRecord(
        question_id=question_id,
        question="Soru?",
        answerable=answerable,
        status=status,
        honest_miss=honest_miss,
        answer_text="Yanıt [S1].",
        citations=("d:1",),
        top_score=12.0,
        n_claims=len(claims) if n_claims is None else n_claims,
        claims=claims,
    )


def provenance() -> dict:
    return {
        "run_id": "run-1",
        "git_commit": "abc123",
        "created_at": datetime(2026, 8, 31, tzinfo=UTC),
        "split": "dev",
        "index_manifest": {"quantization": "int8"},
        "index_revision": "rev/x/int8",
        "calibrator_key": "cal-key",
        "config": {"gate_verifier": True},
        "dataset": {"sha256": "dataset-sha"},
        "budget": {"unit": "api_attempts", "used": 3},
    }


def test_answer_metrics_count_claim_support_completeness_and_false_support():
    records = [
        answer_record(
            question_id="answerable",
            answerable=True,
            verdicts=[("supported", [1]), ("unsupported", [1])],
        ),
        answer_record(
            question_id="false-support",
            answerable=False,
            verdicts=[("supported", [1])],
        ),
        answer_record(
            question_id="honest-miss",
            answerable=False,
            status="answered",
            honest_miss=True,
            verdicts=[],
        ),
    ]

    report = run_answer_eval(records, **provenance())

    assert report.metrics.citation_precision.rate == pytest.approx(2 / 3)
    assert report.metrics.citation_completeness.rate == 1.0
    assert report.metrics.false_supported_answer_rate.numerator == 1
    assert report.metrics.false_supported_answer_rate.denominator == 2
    assert report.metrics.false_supported_answer_rate.rate == 0.5


def test_empty_metric_denominator_is_explicitly_null():
    report = run_answer_eval([], **provenance())

    assert report.metrics.citation_precision.rate is None
    assert report.metrics.citation_precision.upper_bound_95 is None
    assert report.metrics.citation_precision.event_numerator is None
    assert report.metrics.citation_completeness.rate is None
    assert report.metrics.false_supported_answer_rate.rate is None


def test_precision_reports_a_conservative_error_bound_and_lower_bound():
    report = run_answer_eval(
        [answer_record(answerable=True, verdicts=[("supported", [1])] * 3)],
        **provenance(),
    )

    precision = report.metrics.citation_precision
    assert precision.rate == 1.0
    assert precision.event_numerator == 0
    assert precision.error_upper_bound_95 == precision.upper_bound_95
    error_upper = cast(float, precision.error_upper_bound_95)
    assert precision.lower_bound_95 == pytest.approx(1.0 - error_upper)
    assert 0.0 < error_upper < 1.0


def test_completeness_denominator_includes_segmented_but_unverified_claims():
    report = run_answer_eval(
        [
            answer_record(
                answerable=True,
                verdicts=[("supported", [1])],
                n_claims=2,
            )
        ],
        **provenance(),
    )

    completeness = report.metrics.citation_completeness
    assert completeness.numerator == 1
    assert completeness.denominator == 2
    assert completeness.event_numerator == 1
    assert completeness.rate == 0.5


@pytest.mark.parametrize(
    "status,honest_miss,verdicts,expected",
    [
        ("answered", False, [("supported", [1])], 1),
        ("answered", True, [("supported", [1])], 0),
        ("answered", False, [("supported", [1]), ("unsupported", [1])], 0),
        ("abstained", False, [("supported", [1])], 0),
        ("answered", False, [], 0),
    ],
)
def test_false_supported_answer_requires_an_actual_fully_supported_answer(
    status: AnswerStatus,
    honest_miss: bool,
    verdicts: list[tuple[ClaimVerdict, list[int]]],
    expected: int,
):
    report = run_answer_eval(
        [
            answer_record(
                answerable=False,
                status=status,
                honest_miss=honest_miss,
                verdicts=verdicts,
            )
        ],
        **provenance(),
    )

    metric = report.metrics.false_supported_answer_rate
    assert metric.numerator == expected
    assert metric.denominator == 1
    assert metric.upper_bound_95 is not None


def test_report_round_trips_provenance_and_records_as_json(tmp_path):
    report = run_answer_eval(
        [answer_record(answerable=True, verdicts=[("supported", [1])])],
        **provenance(),
    )
    out = tmp_path / "answer-report.json"

    report.to_json(out)

    loaded = report.model_validate_json(out.read_text(encoding="utf-8"))
    assert loaded == report
    assert loaded.index_revision == "rev/x/int8"
    assert loaded.dataset["sha256"] == "dataset-sha"
