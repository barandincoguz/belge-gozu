import math

import pytest

from belge_gozu.answer.late_calibrate import (
    LATE_FEATURE_ORDER,
    LateCalibrationArtifact,
    LateCalibrationMismatch,
    LateCalibrationRow,
    assign_inner_split,
    enablement_verdict,
    evaluate_late_calibration,
    fit_late_calibration,
    group_key,
    late_calibration_key,
    validate_partition,
)


def _raw(**updates):
    row = {
        "question_id": "c1",
        "answerable": True,
        "gold_doc_ids": ["k4721"],
        "unanswerable_reason": None,
        "slice": "paraphrase",
    }
    row.update(updates)
    return row


def _row(qid: str, label: int, **feature_updates) -> LateCalibrationRow:
    features = {
        "mogan_top1_mean": 0.8,
        "mogan_margin_mean": 0.1,
        "colmm_top1_mean": 0.7,
        "colmm_margin_mean": 0.05,
        **feature_updates,
    }
    return LateCalibrationRow(
        question_id=qid,
        outer_split="dev",
        inner_split="fit",
        group="doc:k4721",
        answerable=bool(label),
        label=label,
        gold_in_topk=bool(label),
        unanswerable_reason=None if label else "korpus-disi",
        slice="paraphrase" if label else "korpus-disi",
        source="canary" if label else "unans",
        features=features,
        diagnostics={"mogan_raw_top1": 16.0, "mogan_query_tokens": 20.0},
    )


def test_late_feature_order_is_small_and_bm25_independent():
    assert LATE_FEATURE_ORDER == (
        "mogan_top1_mean",
        "mogan_margin_mean",
        "colmm_top1_mean",
        "colmm_margin_mean",
    )


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (_raw(), "doc:k4721"),
        (
            _raw(
                question_id="u1",
                answerable=False,
                gold_doc_ids=[],
                unanswerable_reason="korpus-disi",
                _anchor_law="5901",
            ),
            "anchor:5901",
        ),
        (
            _raw(
                question_id="u2",
                answerable=False,
                gold_doc_ids=[],
                unanswerable_reason="eksik-kanit",
                _subject_doc="k4857",
            ),
            "doc:k4857",
        ),
        (
            _raw(
                question_id="u201",
                answerable=False,
                gold_doc_ids=[],
                unanswerable_reason="anlamsiz-ood",
            ),
            "qid:u201",
        ),
    ],
)
def test_group_key_follows_the_law_grouping_contract(row, expected):
    assert group_key(row) == expected


def test_inner_split_is_deterministic_and_keeps_a_group_together():
    a = _raw(question_id="c1", gold_doc_ids=["k4721"])
    b = _raw(question_id="c2", gold_doc_ids=["k4721"])
    assert assign_inner_split(a) == assign_inner_split(a)
    assert assign_inner_split(a) == assign_inner_split(b)
    assert assign_inner_split(a) in {"fit", "calibration"}


def test_partition_requires_both_labels():
    with pytest.raises(ValueError, match="iki sınıf"):
        validate_partition([_row("p1", 1)], name="fit")


def test_partition_rejects_missing_or_non_finite_features():
    missing = _row("n", 0)
    missing.features.pop("colmm_margin_mean")
    with pytest.raises(ValueError, match="özellik anahtarları"):
        validate_partition([_row("p", 1), missing], name="fit")

    bad = _row("n", 0)
    bad.features["mogan_top1_mean"] = math.nan
    with pytest.raises(ValueError, match="sonlu"):
        validate_partition([_row("p", 1), bad], name="fit")


def _separable_rows(inner_split: str) -> list[LateCalibrationRow]:
    rows = []
    for i, value in enumerate((0.90, 0.85, 0.20, 0.10)):
        label = int(i < 2)
        row = _row(
            f"{inner_split}-{i}",
            label,
            mogan_top1_mean=value,
            mogan_margin_mean=value / 10,
            colmm_top1_mean=value - 0.02,
            colmm_margin_mean=value / 20,
        )
        object.__setattr__(row, "inner_split", inner_split)
        rows.append(row)
    return rows


def _identity():
    return {
        "index_revision": "rev/traincompat/int8",
        "bm25_recipe_fingerprint": "abc123",
        "late_recipe": "late-score-v1",
        "channels": [
            {"name": "mogan", "revision": "m1", "sidecar_sha256": "1" * 64},
            {"name": "colmm", "revision": "c1", "sidecar_sha256": "2" * 64},
        ],
    }


def test_artifact_round_trip_preserves_model_threshold_and_identity(tmp_path):
    artifact = fit_late_calibration(
        _separable_rows("fit"),
        _separable_rows("calibration"),
        identity=_identity(),
    )

    path = artifact.save(tmp_path / "late")
    loaded = LateCalibrationArtifact.load(path, expected_key=artifact.key)

    assert loaded.tau == artifact.tau
    assert loaded.identity == artifact.identity
    assert loaded.calibrator.to_dict() == artifact.calibrator.to_dict()
    assert not list(path.parent.glob("*.tmp"))


def test_artifact_load_refuses_a_different_runtime_identity(tmp_path):
    artifact = fit_late_calibration(
        _separable_rows("fit"),
        _separable_rows("calibration"),
        identity=_identity(),
    )
    path = artifact.save(tmp_path)

    changed = {**_identity(), "late_recipe": "late-score-v2"}
    with pytest.raises(LateCalibrationMismatch, match="anahtarı uyuşmuyor"):
        LateCalibrationArtifact.load(path, expected_key=late_calibration_key(changed))


def test_evaluation_never_refits_and_reports_operating_point():
    artifact = fit_late_calibration(
        _separable_rows("fit"),
        _separable_rows("calibration"),
        identity=_identity(),
    )

    metrics = evaluate_late_calibration(artifact, _separable_rows("calibration"))

    assert metrics["counts"] == {"total": 4, "positive": 2, "negative": 2}
    assert metrics["tau"] == artifact.tau
    assert 0.0 <= metrics["coverage_at_tau"] <= 1.0
    assert metrics["safe_answerable_accept"]["n"] == 2
    assert metrics["false_answer_on_unanswerable"]["n"] == 2
    assert len(metrics["per_question"]) == 4


def test_enablement_verdict_requires_all_five_checks():
    passing = {
        "risk_at_tau": 0.04,
        "safe_answerable_accept": {"rate": 0.80},
        "false_answer_on_unanswerable": {"rate": 0.01, "upper_bound_95": 0.04},
    }
    verdict = enablement_verdict(passing, identity_matches=True)
    assert verdict["eligible_to_enable"] is True
    assert all(verdict["checks"].values())

    failing = {**passing, "safe_answerable_accept": {"rate": 0.79}}
    verdict = enablement_verdict(failing, identity_matches=True)
    assert verdict["eligible_to_enable"] is False
    assert verdict["checks"]["safe_answerable_accept_rate_gte_0_80"] is False
