import json
from pathlib import Path

import pytest
from scripts.calibrate_late_channel import (
    ScoredQuestion,
    build_labeled_rows,
    diagnostic_stats,
    require_final_gate,
    run_eval_from_rows,
    run_fit_from_rows,
)

from belge_gozu.answer.late_calibrate import LateCalibrationRow, late_calibration_key

FEATURES = {
    "mogan_top1_mean": 0.8,
    "mogan_margin_mean": 0.1,
    "colmm_top1_mean": 0.7,
    "colmm_margin_mean": 0.05,
}


def _raw(qid: str, doc: str, *, answerable: bool = True):
    return {
        "question_id": qid,
        "question": f"{qid} sorusu",
        "answerable": answerable,
        "gold_doc_ids": [doc] if answerable else [],
        "gold_page_ids": [f"{doc}:1"] if answerable else [],
        "unanswerable_reason": None if answerable else "anlamsiz-ood",
        "slice": "paraphrase" if answerable else "anlamsiz-ood",
    }


def _late_row(qid: str, label: int, inner: str, value: float) -> LateCalibrationRow:
    return LateCalibrationRow(
        question_id=qid,
        outer_split="dev" if inner != "test" else "test",
        inner_split=inner,
        group=f"qid:{qid}",
        answerable=bool(label),
        label=label,
        gold_in_topk=bool(label),
        unanswerable_reason=None if label else "anlamsiz-ood",
        slice="paraphrase" if label else "anlamsiz-ood",
        source="retrieval_eval" if label else "abstention_eval",
        features={
            "mogan_top1_mean": value,
            "mogan_margin_mean": value / 10,
            "colmm_top1_mean": value - 0.01,
            "colmm_margin_mean": value / 20,
        },
        diagnostics={
            "mogan_raw_top1": value * 20,
            "mogan_query_tokens": 20.0,
            "colmm_raw_top1": value * 32,
            "colmm_query_tokens": 32.0,
        },
    )


def _identity():
    return {
        "index_revision": "rev/int8",
        "bm25_recipe_fingerprint": "abc123",
        "late_recipe": "late-score-v1",
        "channels": [{"slot": "mogan"}, {"slot": "colmm"}],
    }


def test_build_labeled_rows_filters_before_scoring_and_uses_union_top5():
    raw = [_raw("dev-hit", "k1"), _raw("test-never-scored", "k2")]
    splits = {"dev_docs": {"k1"}, "test_docs": {"k2"}}
    calls = []

    def scorer(rec):
        calls.append(rec["question_id"])
        return ScoredQuestion(
            ranking=("x:1", "x:2", "k1:1"),
            features=FEATURES,
            diagnostics={"mogan_raw_top1": 16.0, "mogan_query_tokens": 20.0},
        )

    rows = build_labeled_rows(raw, splits, outer_split="dev", scorer=scorer)

    assert calls == ["dev-hit"]
    assert len(rows) == 1
    assert rows[0].label == 1 and rows[0].gold_in_topk is True
    assert rows[0].outer_split == "dev"
    assert rows[0].inner_split in {"fit", "calibration"}


def test_eval_requires_an_explicit_one_shot_gate():
    with pytest.raises(SystemExit, match="--yes-final-gate"):
        require_final_gate(False)
    require_final_gate(True)


def test_diagnostics_expose_raw_length_bias_and_normalized_auc():
    rows = [
        _late_row("p1", 1, "fit", 0.9),
        _late_row("p2", 1, "fit", 0.8),
        _late_row("n1", 0, "fit", 0.2),
        _late_row("n2", 0, "fit", 0.1),
    ]
    stats = diagnostic_stats(rows)
    assert stats["mogan_top1_mean"]["auc"] == pytest.approx(1.0)
    assert "query_token_correlation" in stats["mogan_raw_top1"]


def test_fit_and_eval_reports_share_the_locked_artifact(tmp_path: Path):
    fit = [_late_row("fp1", 1, "fit", 0.9), _late_row("fn1", 0, "fit", 0.1)]
    calibration = [
        _late_row("cp1", 1, "calibration", 0.85),
        _late_row("cn1", 0, "calibration", 0.15),
    ]
    artifact_dir = tmp_path / "artifact"
    fit_out = tmp_path / "fit.json"

    fit_report = run_fit_from_rows(
        [*fit, *calibration],
        identity=_identity(),
        data_kunye={"data_files": []},
        artifact_dir=artifact_dir,
        out=fit_out,
    )

    assert json.loads(fit_out.read_text(encoding="utf-8"))["outer_split"] == "dev"
    assert fit_report["artifact_key"] == late_calibration_key(_identity())
    assert fit_report["threshold"]["selected_on"] == "dev.calibration"

    test_rows = [
        _late_row("tp1", 1, "test", 0.88),
        _late_row("tn1", 0, "test", 0.12),
    ]
    eval_out = tmp_path / "eval.json"
    eval_report = run_eval_from_rows(
        test_rows,
        identity=_identity(),
        artifact_dir=artifact_dir,
        out=eval_out,
    )

    assert json.loads(eval_out.read_text(encoding="utf-8"))["outer_split"] == "test"
    assert eval_report["artifact_key"] == fit_report["artifact_key"]
    assert set(eval_report["verdict"]["checks"]) == {
        "selective_risk_point_lte_0_05",
        "unanswerable_false_accept_point_lte_0_02",
        "unanswerable_false_accept_cp95_lte_0_05",
        "safe_answerable_accept_rate_gte_0_80",
        "identity_matches",
    }
