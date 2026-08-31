"""Answer-level G2 metrics with explicit provenance and finite-sample bounds.

This module is deliberately pure: it consumes already-produced answer/gate records,
performs no retrieval or model calls, and can therefore be tested without a network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from belge_gozu.bench.calibration_metrics import clopper_pearson_upper_bound


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ClaimRecord(FrozenModel):
    claim_id: str
    verdict: Literal["supported", "unsupported", "belirsiz"]
    gerekce: str = ""
    cited_sources: tuple[int, ...] = ()
    inherited_sources: bool = False
    cached: bool = False
    attempts: int = Field(default=0, ge=0)


class AnswerRecord(FrozenModel):
    question_id: str
    question: str
    answerable: bool
    unanswerable_reason: str | None = None
    slice: str | None = None
    status: Literal["answered", "abstained", "degraded"]
    honest_miss: bool
    answer_text: str
    citations: tuple[str, ...] = ()
    top_score: float | None = None
    gate1: Mapping[str, Any] | None = None
    n_claims: int = Field(ge=0)
    claims: tuple[ClaimRecord, ...] = ()

    @model_validator(mode="after")
    def _verified_claims_cannot_exceed_segmented_claims(self) -> AnswerRecord:
        if len(self.claims) > self.n_claims:
            raise ValueError(
                "doğrulanmış iddia sayısı bölümlenmiş iddia sayısını aşamaz: "
                f"{len(self.claims)} > {self.n_claims}"
            )
        return self


class RateEstimate(FrozenModel):
    """A rate plus a one-sided 95% upper bound on its safety event/error.

    ``numerator / denominator`` always describes the named metric. For quality
    metrics (precision/completeness), ``event_numerator`` is the complementary
    error count used for ``upper_bound_95``. For an error metric, the two
    numerators are identical. Empty denominators are represented by ``None``.
    """

    numerator: int
    denominator: int
    rate: float | None
    event_numerator: int | None
    upper_bound_95: float | None
    error_upper_bound_95: float | None = None
    lower_bound_95: float | None = None


class AnswerMetrics(FrozenModel):
    citation_precision: RateEstimate
    citation_completeness: RateEstimate
    false_supported_answer_rate: RateEstimate


class AnswerEvalReport(FrozenModel):
    run_id: str
    git_commit: str
    created_at: datetime
    split: Literal["dev", "test"]
    index_manifest: Mapping[str, Any] | None
    index_revision: str | None
    calibrator_key: str | None
    config: Mapping[str, Any]
    dataset: Mapping[str, Any]
    budget: Mapping[str, Any]
    metrics: AnswerMetrics
    records: tuple[AnswerRecord, ...]

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1) + "\n", encoding="utf-8")


def _quality_estimate(successes: int, total: int, *, precision: bool = False) -> RateEstimate:
    if total == 0:
        return RateEstimate(
            numerator=successes,
            denominator=0,
            rate=None,
            event_numerator=None,
            upper_bound_95=None,
            error_upper_bound_95=None,
            lower_bound_95=None,
        )
    errors = total - successes
    error_upper = clopper_pearson_upper_bound(errors, total)
    return RateEstimate(
        numerator=successes,
        denominator=total,
        rate=successes / total,
        event_numerator=errors,
        upper_bound_95=error_upper,
        error_upper_bound_95=error_upper if precision else None,
        lower_bound_95=(1.0 - error_upper) if precision else None,
    )


def _event_estimate(events: int, total: int) -> RateEstimate:
    if total == 0:
        return RateEstimate(
            numerator=events,
            denominator=0,
            rate=None,
            event_numerator=None,
            upper_bound_95=None,
        )
    return RateEstimate(
        numerator=events,
        denominator=total,
        rate=events / total,
        event_numerator=events,
        upper_bound_95=clopper_pearson_upper_bound(events, total),
    )


def _metrics(records: Sequence[AnswerRecord]) -> AnswerMetrics:
    claims = [claim for record in records for claim in record.claims]
    supported = sum(claim.verdict == "supported" for claim in claims)

    segmented = sum(record.n_claims for record in records)
    cited = sum(bool(claim.cited_sources) for claim in claims)

    unanswerable = [record for record in records if not record.answerable]
    false_supported = sum(
        record.status == "answered"
        and not record.honest_miss
        and bool(record.claims)
        and all(claim.verdict == "supported" for claim in record.claims)
        for record in unanswerable
    )

    return AnswerMetrics(
        citation_precision=_quality_estimate(supported, len(claims), precision=True),
        citation_completeness=_quality_estimate(cited, segmented),
        false_supported_answer_rate=_event_estimate(false_supported, len(unanswerable)),
    )


def run_answer_eval(
    records: Sequence[AnswerRecord],
    *,
    run_id: str,
    git_commit: str,
    created_at: datetime,
    split: Literal["dev", "test"],
    index_manifest: Mapping[str, Any] | None,
    index_revision: str | None,
    calibrator_key: str | None,
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> AnswerEvalReport:
    """Compute G2 metrics from immutable answer records without external I/O."""
    frozen_records = tuple(records)
    return AnswerEvalReport(
        run_id=run_id,
        git_commit=git_commit,
        created_at=created_at,
        split=split,
        index_manifest=index_manifest,
        index_revision=index_revision,
        calibrator_key=calibrator_key,
        config=config,
        dataset=dataset,
        budget=budget,
        metrics=_metrics(frozen_records),
        records=frozen_records,
    )
