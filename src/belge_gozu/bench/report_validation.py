"""Deterministic validation for persisted benchmark reports.

The validator consumes report JSON only. It never loads a model or calls a
network, and it fails closed when a canonical aggregate cannot be recomputed
from the persisted per-question/per-record evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from belge_gozu.bench.answer_eval import AnswerEvalReport, run_answer_eval
from belge_gozu.bench.dataset import load_bench, load_splits, question_split, select_bench
from belge_gozu.bench.dense_artifacts import sha256_file
from belge_gozu.bench.harness import EvalReport
from belge_gozu.bench.metrics import bootstrap_ci, ndcg_at_k, recall_at_k
from belge_gozu.index.manifest import IndexManifest


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
                actual = sha256_file(target)
                if actual != digest:
                    raise ValueError(f"{path}.sha256 uyuşmuyor: kayıtlı={digest}, yeniden={actual}")
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


def _select_report_bench(
    metadata: Mapping[str, Any], bench_path: str, *, legacy_only_verified: bool
) -> list:
    """Retrieval ve oracle raporlarının aynı seçim/split künyesini doğrula."""
    verification = metadata.get("verification")
    if verification is not None:
        if not isinstance(verification, Mapping) or not isinstance(
            verification.get("only_verified"), bool
        ):
            raise ValueError("config.verification.only_verified bool olmalı")
        selection = select_bench(
            bench_path,
            only_verified=verification["only_verified"],
            min_verification=verification.get("min_verification"),
        )
        if dict(verification) != selection.provenance():
            raise ValueError("config.verification seçimi veri kümesiyle uyuşmuyor")
        questions = selection.questions
    else:
        # Eski raporlar seçimi config köküne yazmıştı; yalnız onlarda bu yolu koru.
        questions = load_bench(
            bench_path,
            only_verified=bool(metadata.get("only_verified", legacy_only_verified)),
            min_verification=metadata.get("min_verification"),
        )
    benchmark = metadata.get("benchmark")
    if benchmark is not None:
        if not isinstance(benchmark, Mapping) or benchmark.get("path") != bench_path:
            raise ValueError("config.benchmark.path ile config.bench uyuşmuyor")
        if benchmark.get("sha256") != sha256_file(Path(bench_path)):
            raise ValueError("config.benchmark.sha256 veri kümesiyle uyuşmuyor")
    split = metadata.get("split")
    if split is not None:
        if split not in ("dev", "test"):
            raise ValueError(f"config.split geçersiz: {split!r}")
        split_meta = metadata.get("splits")
        if not isinstance(split_meta, Mapping) or not isinstance(split_meta.get("path"), str):
            raise ValueError("config.splits.path zorunludur")
        split_path = Path(split_meta["path"])
        if split_meta.get("sha256") != sha256_file(split_path):
            raise ValueError("config.splits.sha256 bölme dosyasıyla uyuşmuyor")
        splits = load_splits(split_path)
        questions = [q for q in questions if question_split(q, splits) == split]
        if metadata.get("selected_after_split") != len(questions):
            raise ValueError("config.selected_after_split bölme seçimiyle uyuşmuyor")
    return questions


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
    questions = _select_report_bench(report.config, bench_path, legacy_only_verified=True)
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
        row = (gold, ranked, 1 / min(ranks) if ranks else 0.0)
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


def validate_oracle_report_payload(
    payload: Mapping[str, Any], *, require_bench: bool = False
) -> Mapping[str, Any]:
    """Recompute visual oracle recalls from persisted per-question gold ranks."""
    _assert_finite(payload)
    if payload.get("retrieval_pipeline") != "exhaustive-visual":
        raise ValueError("oracle retrieval_pipeline=exhaustive-visual olmalı")
    bench_path = payload.get("bench")
    if not isinstance(bench_path, str):
        if require_bench:
            raise ValueError("oracle bench yolu zorunludur")
        return payload
    if require_bench and (not payload.get("run_id") or not payload.get("git_commit")):
        raise ValueError("oracle run_id/git_commit zorunludur")
    questions = _select_report_bench(payload, bench_path, legacy_only_verified=False)
    answerable = [question for question in questions if question.answerable]
    rows = payload.get("per_question")
    if not isinstance(rows, list) or not answerable:
        raise ValueError("oracle per_question cevaplanabilir soru içermeli")
    if [row.get("question_id") if isinstance(row, Mapping) else None for row in rows] != [
        question.question_id for question in answerable
    ]:
        raise ValueError("oracle per_question question_id sırası benchmark ile uyuşmuyor")

    arm_manifests = {
        "binary": IndexManifest.model_validate(payload.get("packed_manifest")),
        "float": IndexManifest.model_validate(payload.get("float_manifest")),
    }
    if payload.get("int8_manifest") is not None:
        arm_manifests["int8"] = IndexManifest.model_validate(payload["int8_manifest"])
    for arm, quantization in (("binary", "sign-1bit"), ("float", "float16"), ("int8", "int8")):
        manifest = arm_manifests.get(arm)
        if manifest is None:
            continue
        if manifest.quantization != quantization:
            raise ValueError(f"oracle {arm} manifest quantization uyuşmuyor")
        for field in IndexManifest.model_fields:
            if field in {"quantization", "built_at", "git_commit"}:
                continue
            if getattr(manifest, field) != getattr(arm_manifests["binary"], field):
                raise ValueError(f"oracle {arm} manifest {field} uyuşmuyor")

    ks = (1, 5, 20, 50, 200)
    measured = {arm: {k: [] for k in ks} for arm in arm_manifests}
    missing: set[str] = set()
    for question, row in zip(answerable, rows, strict=True):
        gold = set(question.gold_page_ids)
        rank_keys: set[str] | None = None
        for arm, manifest in arm_manifests.items():
            ranks = row.get(f"{arm}_rank")
            if not isinstance(ranks, Mapping) or not set(ranks) <= gold:
                raise ValueError(f"oracle {question.question_id}.{arm}_rank gold ile uyuşmuyor")
            if rank_keys is None:
                rank_keys = set(ranks)
                missing.update(gold - rank_keys)
            elif set(ranks) != rank_keys:
                raise ValueError(f"oracle {question.question_id} kollarının gold kapsamı farklı")
            if any(
                type(rank) is not int or rank < 1 or rank > manifest.n_pages
                for rank in ranks.values()
            ) or len(set(ranks.values())) != len(ranks):
                raise ValueError(f"oracle {question.question_id}.{arm}_rank geçersiz sıra")
            for k in ks:
                measured[arm][k].append(sum(rank <= k for rank in ranks.values()) / len(gold))
    if payload.get("missing_gold_pages") != sorted(missing):
        raise ValueError("oracle missing_gold_pages soru kanıtıyla uyuşmuyor")

    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or summary.get("n") != len(answerable):
        raise ValueError("oracle summary.n uyuşmuyor")
    if set(summary) != {"n", *arm_manifests}:
        raise ValueError("oracle summary kolları manifestlerle uyuşmuyor")
    for arm, values in measured.items():
        stored = summary.get(arm)
        if not isinstance(stored, Mapping) or set(stored) != {str(k) for k in ks}:
            raise ValueError(f"oracle summary.{arm} k değerleri uyuşmuyor")
        for k in ks:
            _check_close(f"summary.{arm}.{k}", stored[str(k)], sum(values[k]) / len(answerable))
    return payload


def validate_reranker_report_payload(
    payload: Mapping[str, Any],
    *,
    require_benchmark: bool = False,
) -> Mapping[str, Any]:
    """Validate custom reranker reports that persist per-question rankings."""
    _assert_finite(payload)
    per_question = payload.get("per_question")
    if not isinstance(per_question, list) or not per_question:
        raise ValueError("reranker report per_question sıralamaları zorunludur")
    question_rows = {
        str(row.get("question_id", "")): row for row in per_question if isinstance(row, Mapping)
    }
    if len(question_rows) != len(per_question) or "" in question_rows:
        raise ValueError("reranker per_question question_id değerleri benzersiz olmalı")

    benchmark = payload.get("benchmark")
    benchmark_path = benchmark.get("path") if isinstance(benchmark, Mapping) else None
    if not isinstance(benchmark_path, str):
        if require_benchmark:
            raise ValueError("reranker report benchmark.path yolu zorunludur")
        return payload
    selection = payload.get("selection") or payload.get("dataset", {}).get("selection") or {}
    questions = load_bench(
        benchmark_path,
        only_verified=bool(selection.get("only_verified", True)),
        min_verification=selection.get("min_verification"),
    )
    expected = {q.question_id: q for q in questions if q.answerable}
    if set(expected) != set(question_rows):
        raise ValueError("benchmark answerable question_id kümesi per_question ile uyuşmuyor")

    arms = ("candidate_pool", "pinned", "unpinned")
    arm_rows: dict[str, list[tuple[set[str], list[str], float]]] = {arm: [] for arm in arms}
    by_slice: dict[str, dict[str, list[tuple[set[str], list[str], float]]]] = {
        arm: {} for arm in arms
    }
    for question_id, question in expected.items():
        row = question_rows[question_id]
        if row.get("gold_page_ids") != question.gold_page_ids:
            raise ValueError(f"per_question.{question_id}.gold_page_ids benchmark ile uyuşmuyor")
        rankings = row.get("rankings")
        if not isinstance(rankings, Mapping):
            raise ValueError(f"per_question.{question_id}.rankings zorunludur")
        gold = set(question.gold_page_ids)
        for arm in arms:
            ranking = rankings.get(arm)
            if not isinstance(ranking, list) or not all(isinstance(pid, str) for pid in ranking):
                raise ValueError(f"per_question.{question_id}.rankings.{arm} liste olmalı")
            if len(ranking) != len(set(ranking)):
                raise ValueError(
                    f"per_question.{question_id}.rankings.{arm} yinelenen page_id içeriyor"
                )
            ranks = [index + 1 for index, pid in enumerate(ranking) if pid in gold]
            metric_row = (gold, ranking, 1 / min(ranks) if ranks else 0.0)
            arm_rows[arm].append(metric_row)
            by_slice[arm].setdefault(question.slice, []).append(metric_row)

    for arm in arms:
        node = payload.get(arm)
        if not isinstance(node, Mapping):
            raise ValueError(f"reranker report {arm} bloğu eksik")
        block = node.get("overall")
        if block is None:
            raise ValueError(f"reranker report {arm}.overall bloğu eksik")
        # Reuse the strict Pydantic block shape used by canonical reports.
        from belge_gozu.bench.harness import MetricBlock

        parsed = MetricBlock.model_validate(block)
        _check_metric_block(arm, parsed, arm_rows[arm])
        per_slice = node.get("per_slice") or {}
        if not isinstance(per_slice, Mapping):
            raise ValueError(f"reranker report {arm}.per_slice nesne olmalı")
        for slice_name, slice_block in per_slice.items():
            parsed_slice = MetricBlock.model_validate(slice_block)
            _check_metric_block(
                f"{arm}.per_slice.{slice_name}",
                parsed_slice,
                by_slice[arm].get(str(slice_name), []),
            )

    candidate = payload.get("candidate_pool")
    coverage = candidate.get("coverage") if isinstance(candidate, Mapping) else None
    if not isinstance(coverage, Mapping):
        raise ValueError("candidate_pool.coverage bloğu eksik")
    candidate_rows = arm_rows["candidate_pool"]
    expected_coverage = {
        "overall": sum(recall_at_k(gold, ranked, len(ranked)) for gold, ranked, _ in candidate_rows)
        / len(candidate_rows),
        "per_slice": {
            name: sum(recall_at_k(gold, ranked, len(ranked)) for gold, ranked, _ in rows)
            / len(rows)
            for name, rows in by_slice["candidate_pool"].items()
        },
    }
    _check_close(
        "candidate_pool.coverage.overall",
        float(coverage["overall"]),
        expected_coverage["overall"],
    )
    for name, expected_value in expected_coverage["per_slice"].items():
        _check_close(
            f"candidate_pool.coverage.per_slice.{name}",
            float((coverage.get("per_slice") or {})[name]),
            expected_value,
        )
    return payload
