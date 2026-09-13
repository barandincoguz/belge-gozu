import json
from datetime import UTC, datetime

import pytest

from belge_gozu.bench.answer_eval import AnswerRecord, ClaimRecord, run_answer_eval
from belge_gozu.bench.dataset import BenchQuestion
from belge_gozu.bench.harness import StageRecord, run_retrieval_eval
from belge_gozu.bench.report_validation import (
    validate_answer_report_payload,
    validate_provenance_hashes,
    validate_retrieval_report_payload,
)
from tests.bench.test_dataset import q_dict


def _provenance() -> dict:
    return {
        "run_id": "run-1",
        "git_commit": "abc123",
        "created_at": datetime(2026, 9, 13, tzinfo=UTC),
        "split": "dev",
        "index_manifest": {"quantization": "int8"},
        "index_revision": "rev/x/int8",
        "calibrator_key": None,
        "config": {"recipe_fingerprint": "recipe-test"},
        "dataset": {"sha256": "dataset-sha"},
        "budget": {"unit": "api_attempts", "max_attempts": 2, "used": 1},
    }


def _answer_payload() -> dict:
    report = run_answer_eval(
        [
            AnswerRecord(
                question_id="q1",
                question="Soru?",
                answerable=True,
                status="answered",
                honest_miss=False,
                answer_text="Yanıt [S1].",
                n_claims=1,
                claims=(ClaimRecord(claim_id="c1", verdict="supported", cited_sources=(1,)),),
            )
        ],
        **_provenance(),
    )
    return report.model_dump(mode="json")


def test_answer_report_validator_recomputes_metrics():
    validate_answer_report_payload(_answer_payload())


def test_answer_report_validator_rejects_tampered_metric():
    payload = _answer_payload()
    payload["metrics"]["citation_precision"]["rate"] = 0.0

    with pytest.raises(ValueError, match="citation_precision"):
        validate_answer_report_payload(payload)


def test_answer_report_validator_rejects_stale_recipe_identity():
    with pytest.raises(ValueError, match="recipe_fingerprint"):
        validate_answer_report_payload(
            _answer_payload(), expected_recipe_fingerprint="different-recipe"
        )


def test_provenance_hash_validator_rejects_tampered_source(tmp_path):
    source = tmp_path / "bench.jsonl"
    source.write_text("original\n", encoding="utf-8")
    payload = {
        "dataset": {
            "bench": {
                "path": str(source),
                "sha256": "0" * 64,
            }
        }
    }

    with pytest.raises(ValueError, match="sha256"):
        validate_provenance_hashes(payload)


class _FullRankPipeline:
    name = "full-rank"

    def run(self, question: str):
        ranked = ["x:1", "k4721:4", "x:2"]
        return ranked, [
            StageRecord(
                stage="final",
                gold_ranks={},
                top_ids=ranked[:1],
                top_scores=[1.0],
                latency_ms=1.0,
                full_ranked=ranked,
            )
        ]


def test_retrieval_report_validator_uses_full_gold_rank_and_rejects_tampering(tmp_path):
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps(q_dict()) + "\n",
        encoding="utf-8",
    )
    report = run_retrieval_eval(
        _FullRankPipeline(),
        [BenchQuestion(**q_dict())],
        known_page_ids={"x:1", "x:2", "k4721:4"},
        ks=(1, 5),
        config={"bench": str(bench), "only_verified": True},
    )
    payload = report.model_dump(mode="json")

    validate_retrieval_report_payload(payload, require_bench=True)
    payload["overall"]["recall_at"]["5"] = 0.0
    with pytest.raises(ValueError, match="recall_at.5"):
        validate_retrieval_report_payload(payload, require_bench=True)
