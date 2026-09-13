"""Deterministic validation for persisted benchmark reports.

The validator consumes report JSON only. It never loads a model or calls a
network, and it fails closed when a canonical aggregate cannot be recomputed
from the persisted per-question/per-record evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from belge_gozu.bench.answer_eval import AnswerEvalReport, run_answer_eval
from belge_gozu.bench.dataset import load_bench
from belge_gozu.bench.harness import EvalReport
from belge_gozu.bench.metrics import bootstrap_ci, ndcg_at_k, recall_at_k


def _assert_finite(value: object, path: str = "report") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} sonlu olmayan sayı içeriyor")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _check_close(path: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{path} uyuşmuyor: kayıtlı={actual!r}, yeniden={expected!r}")


def validate_answer_report_payload(
    payload: Mapping[str, Any],
    *,
    expected_recipe_fingerprint: str | None = None,
    require_identity: bool = False,
) -> AnswerEvalReport:
    """Validate and return a canonical answer report."""
    _assert_finite(payload)
    report = AnswerEvalReport.model_validate(payload)
    question_ids = [record.question_id for record in report.records]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("records question_id değerleri yinelenemez")
    if require_identity and (not report.run_id or not report.git_commit):
        raise ValueError("run_id ve git_commit zorunlu künye alanlarıdır")
    if expected_recipe_fingerprint is not None:
        actual = report.config.get("recipe_fingerprint")
        if actual != expected_recipe_fingerprint:
            raise ValueError(
                "recipe_fingerprint uyuşmuyor: "
                f"kayıtlı={actual!r}, beklenen={expected_recipe_fingerprint!r}"
            )

    recomputed = run_answer_eval(
        report.records,
        run_id=report.run_id,
        git_commit=report.git_commit,
        created_at=report.created_at,
        split=report.split,
        index_manifest=report.index_manifest,
        index_revision=report.index_revision,
        calibrator_key=report.calibrator_key,
        config=report.config,
        dataset=report.dataset,
        budget=report.budget,
    )
    for metric_name in (
        "citation_precision",
        "citation_completeness",
        "false_supported_answer_rate",
    ):
        actual = getattr(report.metrics, metric_name)
        expected = getattr(recomputed.metrics, metric_name)
        if actual != expected:
            raise ValueError(f"metrics.{metric_name} canonical records ile uyuşmuyor")

    legacy = payload.get("per_question")
    if isinstance(legacy, list):
        legacy_ids = [str(row.get("question_id", row.get("qid", ""))) for row in legacy]
        if legacy_ids != question_ids:
            raise ValueError("per_question question_id sırası records ile uyuşmuyor")
    return report


def validate_provenance_hashes(payload: Mapping[str, Any], *, root: Path = Path(".")) -> None:
    """Check every nested ``path`` + ``sha256`` provenance pair on disk."""

    def walk(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            candidate_path = value.get("path")
            digest = value.get("sha256")
            if isinstance(candidate_path, str) and isinstance(digest, str):
                target = Path(candidate_path)
                if not target.is_absolute():
                    target = root / target
                if not target.is_file():
                    raise ValueError(f"{path}.path bulunamadı: {target}")
                actual = sha256(target.read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError(
                        f"{path}.sha256 uyuşmuyor: kayıtlı={digest}, yeniden={actual}"
                    )
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload, "report")


def _metric_block_rows(
    rows: Sequence[tuple[set[str], list[str], float]],
    ks: Sequence[int],
) -> dict[str, Any]:
    if not rows:
        return {
            "recall_at": {k: 0.0 for k in ks},
            "mrr": 0.0,
            "ndcg5": 0.0,
            "n": 0,
            "ci_recall5": None,
        }
    recalls = {k: [recall_at_k(gold, ranked, k) for gold, ranked, _ in rows] for k in ks}
    r5 = recalls.get(5, [])
    return {
        "recall_at": {k: sum(values) / len(values) for k, values in recalls.items()},
        "mrr": sum(value for _, _, value in rows) / len(rows),
        "ndcg5": sum(ndcg_at_k(gold, ranked, 5) for gold, ranked, _ in rows) / len(rows),
        "n": len(rows),
        "ci_recall5": bootstrap_ci(r5) if r5 else None,
    }


def _check_metric_block(
    path: str,
    stored: Any,
    rows: Sequence[tuple[set[str], list[str], float]],
) -> None:
    ks = tuple(sorted(stored.recall_at))
    expected = _metric_block_rows(rows, ks)
    if stored.n != expected["n"]:
        raise ValueError(f"{path}.n uyuşmuyor: kayıtlı={stored.n}, yeniden={expected['n']}")
    for k in ks:
        _check_close(
            f"{path}.recall_at.{k}",
            stored.recall_at[k],
            expected["recall_at"][k],
        )
    _check_close(f"{path}.mrr", stored.mrr, expected["mrr"])
    _check_close(f"{path}.ndcg5", stored.ndcg5, expected["ndcg5"])
    if stored.ci_recall5 is not None and expected["ci_recall5"] is not None:
        for index, (actual, wanted) in enumerate(
            zip(stored.ci_recall5, expected["ci_recall5"], strict=True)
        ):
            _check_close(f"{path}.ci_recall5[{index}]", actual, wanted)


def validate_retrieval_report_payload(
    payload: Mapping[str, Any],
    *,
    require_bench: bool = False,
) -> EvalReport:
    """Validate canonical retrieval aggregates against persisted diagnostics."""
    _assert_finite(payload)
    report = EvalReport.model_validate(payload)
    diagnostics = {diagnostic.question_id: diagnostic for diagnostic in report.diagnostics}
    if len(diagnostics) != len(report.diagnostics):
        raise ValueError("diagnostics question_id değerleri yinelenemez")

    bench_path = report.config.get("bench")
    if not isinstance(bench_path, str):
        if require_bench:
            raise ValueError("retrieval report config.bench yolu zorunludur")
        return report
    questions = load_bench(
        bench_path,
        only_verified=bool(report.config.get("only_verified", True)),
        min_verification=report.config.get("min_verification"),
    )
    answerable = {q.question_id: q for q in questions if q.answerable}
    if set(answerable) != set(diagnostics):
        raise ValueError("bench answerable question_id kümesi diagnostics ile uyuşmuyor")

    rows: list[tuple[set[str], list[str], float]] = []
    by_slice: dict[str, list[tuple[set[str], list[str], float]]] = {}
    by_doc: dict[str, list[tuple[set[str], list[str], float]]] = {}
    for question_id, diagnostic in diagnostics.items():
        question = answerable[question_id]
        ranked = diagnostic.final_ranked
        gold = set(question.gold_page_ids)
        final_stage = diagnostic.stages[-1] if diagnostic.stages else None
        rank_map = final_stage.gold_ranks if final_stage is not None else {}
        ranks: list[int] = []
        for gold_page in gold:
            rank = rank_map.get(gold_page)
            if rank is None:
                rank = ranked.index(gold_page) + 1 if gold_page in ranked else None
            if rank == -1:
                raise ValueError(
                    f"diagnostics.{question_id}.gold_ranks.{gold_page} eski -1 sentinel'ı taşıyor"
                )
            if rank is not None:
                ranks.append(rank)
        if not ranks:
            raise ValueError(f"diagnostics.{question_id} gold rank yeniden hesaplanamıyor")
        row = (gold, ranked, 1 / min(ranks))
        rows.append(row)
        by_slice.setdefault(question.slice, []).append(row)
        for doc_id in question.gold_doc_ids:
            by_doc.setdefault(doc_id, []).append(row)

    _check_metric_block("overall", report.overall, rows)
    for name, block in report.per_slice.items():
        _check_metric_block(f"per_slice.{name}", block, by_slice.get(name, []))
    for name, block in report.per_doc.items():
        _check_metric_block(f"per_doc.{name}", block, by_doc.get(name, []))
    return report
