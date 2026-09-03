"""İki ColBERT kanalının çekimserlik kalibrasyonu ve kilitli test kapısı.

Fit yalnız dış `dev` bölmesini okur; onun içindeki hukuk gruplarını ayrıca
`fit`/`calibration` olarak ayırır. `eval`, mevcut artefaktı değiştirmeden dış
`test` bölmesini ölçer ve açık `--yes-final-gate` olmadan başlamaz.

Gerçek koşum örneği:

    uv run python scripts/calibrate_late_channel.py fit \
      --index-dir data/index-traincompat-int8 \
      --late-index data/index-colbert-mogan-f16 \
      --late-index data/index-colbert-colmm-f16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.answer.calibrate import git_blob_sha, sha256_file, univariate_auc  # noqa: E402
from belge_gozu.answer.late_calibrate import (  # noqa: E402
    LATE_RECIPE_ID,
    LateCalibrationArtifact,
    LateCalibrationRow,
    assign_inner_split,
    enablement_verdict,
    evaluate_late_calibration,
    fit_late_calibration,
    group_key,
    late_calibration_key,
)
from belge_gozu.bench.dataset import BenchQuestion, assign_split  # noqa: E402
from belge_gozu.index.colbert_encode import ColBERTEncoder  # noqa: E402
from belge_gozu.index.manifest import index_revision, read_manifest  # noqa: E402
from belge_gozu.provenance import git_commit  # noqa: E402
from belge_gozu.retrieval.hybrid import load_text_channel  # noqa: E402
from belge_gozu.retrieval.late import LateInteractionChannel  # noqa: E402
from belge_gozu.retrieval.text import (  # noqa: E402
    rank_order,
    recipe_fingerprint,
    route_window,
    routed_docs,
)
from belge_gozu.retrieval.union import union_candidates  # noqa: E402

DEFAULT_RETRIEVAL_EVAL = REPO_ROOT / "data/bench/retrieval_eval_v2.jsonl"
DEFAULT_ABSTENTION_EVAL = REPO_ROOT / "data/bench/abstention_eval_v1.jsonl"
DEFAULT_SPLITS = REPO_ROOT / "data/bench/splits_v1.json"
DEFAULT_INDEX = REPO_ROOT / "data/index-traincompat-int8"
DEFAULT_LATE_INDICES = (
    REPO_ROOT / "data/index-colbert-mogan-f16",
    REPO_ROOT / "data/index-colbert-colmm-f16",
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "data/calibration/late-channel-v1"
CHANNEL_SLOTS = ("mogan", "colmm")
TOP_K = 5


@dataclass(frozen=True)
class ScoredQuestion:
    """Üretim sırası ile kalibrasyon/teşhis özelliklerinin tek sorguluk çıktısı."""

    ranking: tuple[str, ...]
    features: Mapping[str, float]
    diagnostics: Mapping[str, float]


def require_final_gate(allowed: bool) -> None:
    if not allowed:
        raise SystemExit(
            "kilitli test bölmesi yalnız faz-sonu kapısında okunabilir; "
            "bilinçli koşum için --yes-final-gate verin"
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def load_benchmark_rows(retrieval_eval: Path, abstention_eval: Path) -> list[dict[str, Any]]:
    """Dondurulmuş iki kaynaktan yalnız tasarımda izin verilen satırları al."""
    selected: list[dict[str, Any]] = []
    for source, path in (("retrieval_eval", retrieval_eval), ("abstention_eval", abstention_eval)):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                BenchQuestion(**row)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: geçersiz bench satırı: {exc}") from exc
            if source == "retrieval_eval":
                if not (row.get("answerable") and row.get("verification_kind") == "human"):
                    continue
            elif row.get("verification_status") != "verified":
                continue
            selected.append({**row, "_calibration_source": source})
    if not selected:
        raise ValueError("kalibrasyon için uygun bench satırı bulunamadı")
    return selected


def load_splits(path: Path) -> dict[str, set[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dev_docs": set(raw.get("dev_docs") or []),
        "test_docs": set(raw.get("test_docs") or []),
    }


def build_labeled_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    splits: dict[str, set[str]],
    *,
    outer_split: Literal["dev", "test"],
    scorer: Callable[[Mapping[str, Any]], ScoredQuestion],
    top_k: int = TOP_K,
) -> list[LateCalibrationRow]:
    """Dış yakayı SKORLAMADAN ÖNCE filtrele ve top-k dayanak etiketini kur."""
    out: list[LateCalibrationRow] = []
    for raw in raw_rows:
        assigned = assign_split(raw, splits)
        if assigned != outer_split:
            continue
        scored = scorer(raw)
        answerable = bool(raw["answerable"])
        gold = set(raw.get("gold_page_ids") or [])
        gold_in_topk = answerable and bool(gold & set(scored.ranking[:top_k]))
        out.append(
            LateCalibrationRow(
                question_id=str(raw["question_id"]),
                outer_split=outer_split,
                inner_split=assign_inner_split(raw) if outer_split == "dev" else "test",
                group=group_key(raw),
                answerable=answerable,
                label=int(gold_in_topk),
                gold_in_topk=gold_in_topk,
                unanswerable_reason=raw.get("unanswerable_reason"),
                slice=str(raw.get("slice") or "unknown"),
                source=str(
                    raw.get("_calibration_source")
                    or ("retrieval_eval" if answerable else "abstention_eval")
                ),
                features={key: float(value) for key, value in scored.features.items()},
                diagnostics={key: float(value) for key, value in scored.diagnostics.items()},
            )
        )
    if not out:
        raise ValueError(f"{outer_split} dış bölmesinde satır yok")
    return out


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def diagnostic_stats(rows: Sequence[LateCalibrationRow]) -> dict[str, dict[str, Any]]:
    """Ham-vs-normalize deney kanıtı; model seçimi veya tau için kullanılmaz."""
    labels = np.array([row.label for row in rows], dtype=np.float64)
    if set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("teşhis AUC'si için iki sınıf da gerekli")
    out: dict[str, dict[str, Any]] = {}
    for slot in CHANNEL_SLOTS:
        q_tokens = np.array(
            [row.diagnostics[f"{slot}_query_tokens"] for row in rows], dtype=np.float64
        )
        for field, source in (
            (f"{slot}_raw_top1", "diagnostics"),
            (f"{slot}_top1_mean", "features"),
        ):
            values = np.array([getattr(row, source)[field] for row in rows], dtype=np.float64)
            out[field] = {
                "auc": univariate_auc(values, labels),
                "query_token_correlation": _correlation(values, q_tokens),
                "positive_median": float(np.median(values[labels == 1.0])),
                "negative_median": float(np.median(values[labels == 0.0])),
            }
    return out


def run_fit_from_rows(
    rows: Sequence[LateCalibrationRow],
    *,
    identity: Mapping[str, Any],
    data_kunye: Mapping[str, Any],
    artifact_dir: Path,
    out: Path,
) -> dict[str, Any]:
    fit_rows = [row for row in rows if row.inner_split == "fit"]
    calibration_rows = [row for row in rows if row.inner_split == "calibration"]
    artifact = fit_late_calibration(
        fit_rows,
        calibration_rows,
        identity=identity,
        data_kunye=data_kunye,
    )
    artifact_path = artifact.save(artifact_dir)
    report = {
        "schema_version": 1,
        "mode": "fit",
        "outer_split": "dev",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "artifact_key": artifact.key,
        "artifact_path": str(artifact_path),
        "identity": artifact.identity,
        "threshold": {**artifact.threshold, "selected_on": "dev.calibration"},
        "counts": {
            "dev_total": len(rows),
            "fit": artifact.kunye["fit_counts"],
            "calibration": artifact.kunye["calibration_counts"],
        },
        "diagnostics": diagnostic_stats(rows),
        "fit_metrics": artifact.kunye["fit_metrics"],
        "calibration_metrics": artifact.kunye["calibration_metrics"],
        "data_kunye": dict(data_kunye),
    }
    _atomic_json(out, report)
    return report


def run_eval_from_rows(
    rows: Sequence[LateCalibrationRow],
    *,
    identity: Mapping[str, Any],
    artifact_dir: Path,
    out: Path,
) -> dict[str, Any]:
    expected_key = late_calibration_key(identity)
    artifact = LateCalibrationArtifact.load(artifact_dir, expected_key=expected_key)
    metrics = evaluate_late_calibration(artifact, rows)
    verdict = enablement_verdict(metrics, identity_matches=artifact.identity == dict(identity))
    report = {
        "schema_version": 1,
        "mode": "eval",
        "outer_split": "test",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "artifact_key": artifact.key,
        "identity": artifact.identity,
        "threshold": artifact.threshold,
        "metrics": metrics,
        "diagnostics": diagnostic_stats(rows),
        "verdict": verdict,
    }
    _atomic_json(out, report)
    return report


@dataclass(frozen=True)
class _LoadedChannel:
    slot: str
    path: Path
    sidecar: dict[str, Any]
    channel: LateInteractionChannel


class ProductionScorer:
    """Bench ile sevk kodunun aynı sınıfları/formülü kullanmasını sağlar."""

    def __init__(self, index_dir: Path, late_indices: Sequence[Path]):
        if len(late_indices) != len(CHANNEL_SLOTS):
            raise ValueError(
                f"tam iki --late-index gerekli ({CHANNEL_SLOTS}); gelen={len(late_indices)}"
            )
        chunks = pd.read_parquet(index_dir / "chunks.parquet")
        self.chunk_pages = {
            str(chunk_id): tuple(str(pid) for pid in page_ids)
            for chunk_id, page_ids in zip(
                chunks["chunk_id"].tolist(), chunks["page_ids"].tolist(), strict=True
            )
        }
        pages = pd.read_parquet(index_dir / "page_texts.parquet")
        self.page_ids = [str(pid) for pid in pages.page_id.tolist()]
        self.bm25, self.doc_names = load_text_channel(index_dir, self.page_ids)
        self.channels = tuple(
            self._load_channel(slot, Path(path))
            for slot, path in zip(CHANNEL_SLOTS, late_indices, strict=True)
        )
        self.index_dir = index_dir

    def _load_channel(self, slot: str, path: Path) -> _LoadedChannel:
        sidecar_path = path / "colbert.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        repo_name = str(sidecar["model_repo"]).lower()
        if slot not in repo_name:
            raise ValueError(
                f"{slot} kanal yuvası yanlış modelle eşleşti: {sidecar['model_repo']} ({path})"
            )
        channel = LateInteractionChannel(
            embeddings=np.load(path / "embs.npy", mmap_mode="r"),
            offsets=np.load(path / "offsets.npy"),
            chunk_ids=json.loads((path / "chunk_ids.json").read_text(encoding="utf-8")),
            chunk_pages=self.chunk_pages,
            encoder=ColBERTEncoder(
                str(sidecar["model_repo"]),
                str(sidecar["revision"]),
                document_length=int(sidecar["document_length"]),
            ),
        )
        return _LoadedChannel(slot=slot, path=path, sidecar=sidecar, channel=channel)

    def __call__(self, row: Mapping[str, Any]) -> ScoredQuestion:
        query = str(row["question"])
        bm25_scores = self.bm25.scores(query)
        order = rank_order(bm25_scores)
        ranking = route_window(
            [self.page_ids[int(i)] for i in order],
            routed_docs(query, self.doc_names),
        )
        features: dict[str, float] = {}
        diagnostics: dict[str, float] = {}
        for loaded in self.channels:
            result = loaded.channel.search_with_scores(query)
            ranking = union_candidates(ranking, list(result.pages))
            features[f"{loaded.slot}_top1_mean"] = result.mean_top1
            features[f"{loaded.slot}_margin_mean"] = result.mean_margin
            diagnostics[f"{loaded.slot}_raw_top1"] = result.raw_top1
            diagnostics[f"{loaded.slot}_raw_margin"] = result.raw_margin
            diagnostics[f"{loaded.slot}_query_tokens"] = float(result.query_tokens)
        return ScoredQuestion(
            ranking=tuple(ranking),
            features=features,
            diagnostics=diagnostics,
        )

    def identity(self) -> dict[str, Any]:
        manifest = read_manifest(self.index_dir)
        if manifest is None:
            raise ValueError(f"ana indeks manifesti yok: {self.index_dir}")
        channels = []
        for loaded in self.channels:
            hashes = {
                name: sha256_file(loaded.path / name)
                for name in ("colbert.json", "chunk_ids.json", "offsets.npy", "embs.npy")
            }
            channels.append(
                {
                    "slot": loaded.slot,
                    "model_repo": str(loaded.sidecar["model_repo"]),
                    "revision": str(loaded.sidecar["revision"]),
                    "document_length": int(loaded.sidecar["document_length"]),
                    "query_length": int(loaded.sidecar["query_length"]),
                    "files_sha256": hashes,
                }
            )
        return {
            "index_revision": index_revision(manifest),
            "bm25_recipe_fingerprint": recipe_fingerprint(),
            "late_recipe": LATE_RECIPE_ID,
            "channels": channels,
        }


def _data_kunye(retrieval_eval: Path, abstention_eval: Path, splits: Path) -> dict[str, Any]:
    files = []
    for name, path in (
        ("retrieval_eval", retrieval_eval),
        ("abstention_eval", abstention_eval),
        ("splits", splits),
    ):
        files.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "git_blob": git_blob_sha(path),
            }
        )
    return {
        "data_files": files,
        "answerable_filter": 'answerable=true AND verification_kind="human"',
        "unanswerable_filter": 'verification_status="verified"',
        "abstention_eval_verification_caveat": (
            "abstention_eval_v1 büyük ölçüde model çapraz-kontrollü veya mekanik doğrulanmıştır; "
            "insan-doğrulanmış diye yorumlanamaz"
        ),
    }


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval-eval", type=Path, default=DEFAULT_RETRIEVAL_EVAL)
    parser.add_argument("--abstention-eval", type=Path, default=DEFAULT_ABSTENTION_EVAL)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--late-index", type=Path, action="append", dest="late_indices")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="yalnız dev üzerinde fit + eşik seçimi")
    _add_common(fit)
    evaluate = sub.add_parser("eval", help="kilitli test bölmesinde tek koşum")
    _add_common(evaluate)
    evaluate.add_argument("--yes-final-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "eval":
        require_final_gate(args.yes_final_gate)
    late_indices = tuple(args.late_indices or DEFAULT_LATE_INDICES)
    raw_rows = load_benchmark_rows(args.retrieval_eval, args.abstention_eval)
    splits = load_splits(args.splits)
    scorer = ProductionScorer(args.index_dir, late_indices)
    identity = scorer.identity()
    outer: Literal["dev", "test"] = "dev" if args.command == "fit" else "test"
    rows = build_labeled_rows(raw_rows, splits, outer_split=outer, scorer=scorer)
    if args.command == "fit":
        report = run_fit_from_rows(
            rows,
            identity=identity,
            data_kunye=_data_kunye(args.retrieval_eval, args.abstention_eval, args.splits),
            artifact_dir=args.artifact_dir,
            out=args.out,
        )
        print(
            f"dev n={report['counts']['dev_total']} tau={report['threshold']['value']:.6f} "
            f"artifact={report['artifact_key']} -> {args.out}"
        )
    else:
        report = run_eval_from_rows(
            rows,
            identity=identity,
            artifact_dir=args.artifact_dir,
            out=args.out,
        )
        status = "ELIGIBLE" if report["verdict"]["eligible_to_enable"] else "BLOCKED"
        print(f"test n={report['metrics']['counts']['total']} {status} -> {args.out}")
        for name, passed in report["verdict"]["checks"].items():
            print(f"  {'PASS' if passed else 'FAIL'} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
