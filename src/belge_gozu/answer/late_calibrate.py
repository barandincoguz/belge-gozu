"""Geç-etkileşim kanalının ayrı, sızıntı-dirençli çekimserlik kalibrasyonu.

BM25 kalibratörü `answer/calibrate.py` içinde kalır. Bu modül onun deterministik
NumPy lojistik modelini yeniden kullanır fakat özellik şemasını ve artefakt
kimliğini ayrı tutar; böylece donmuş BM25 reçetesi geç-kanal deneyinin nesnesi
olmaz.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from belge_gozu.answer.calibrate import Calibrator, choose_threshold
from belge_gozu.bench.calibration_metrics import (
    auroc,
    brier,
    ece,
    false_answer_rate_on_unanswerable,
    risk_coverage,
    risk_coverage_auc,
)

LATE_FEATURE_ORDER: tuple[str, ...] = (
    "mogan_top1_mean",
    "mogan_margin_mean",
    "colmm_top1_mean",
    "colmm_margin_mean",
)

INNER_SPLIT_SALT = "late-calibration-v1"
LATE_RECIPE_ID = "late-score-v1:content-token-mean+page-max+sequential-interleave"
LATE_ARTIFACT_FILENAME = "calibrator.json"
LATE_ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_MAX_RISK = 0.05


class LateCalibrationMismatch(RuntimeError):
    """Artefakt, çalışma anındaki geç-kanal kimliğiyle uyuşmuyor."""


@dataclass(frozen=True)
class LateCalibrationRow:
    """Bir bench sorusunun üretim geçişinden çıkarılan kalibrasyon satırı."""

    question_id: str
    outer_split: str
    inner_split: str
    group: str
    answerable: bool
    label: int
    gold_in_topk: bool
    unanswerable_reason: str | None
    slice: str
    source: str
    features: dict[str, float]
    diagnostics: dict[str, float]


def group_key(row: Mapping[str, Any]) -> str:
    """Mevcut dış bölmenin hukuk-gruplu birimini iç bölmeye de taşır."""
    gold_docs = row.get("gold_doc_ids") or []
    if gold_docs:
        return f"doc:{gold_docs[0]}"
    reason = row.get("unanswerable_reason")
    if reason == "korpus-disi" and row.get("_anchor_law"):
        return f"anchor:{row['_anchor_law']}"
    if reason == "eksik-kanit" and row.get("_subject_doc"):
        return f"doc:{row['_subject_doc']}"
    question_id = str(row.get("question_id") or "")
    if not question_id:
        raise ValueError("iç bölme için question_id gerekli")
    return f"qid:{question_id}"


def assign_inner_split(row: Mapping[str, Any]) -> Literal["fit", "calibration"]:
    """Dev satırını deterministik 2/3 fit, 1/3 calibration yakasına atar."""
    payload = f"{INNER_SPLIT_SALT}:{group_key(row)}".encode()
    return "fit" if hashlib.sha256(payload).digest()[0] < 170 else "calibration"


def validate_partition(rows: Sequence[LateCalibrationRow], *, name: str) -> None:
    """Bir fit/eşik yakasının sınıf ve özellik sözleşmesini doğrular."""
    if not rows:
        raise ValueError(f"{name} bölmesi boş")
    labels = {int(row.label) for row in rows}
    if labels != {0, 1}:
        raise ValueError(f"{name} bölmesinde iki sınıf da gerekli; bulunan={sorted(labels)}")
    expected = set(LATE_FEATURE_ORDER)
    for row in rows:
        if set(row.features) != expected:
            raise ValueError(
                f"{row.question_id}: özellik anahtarları uyuşmuyor; "
                f"beklenen={list(LATE_FEATURE_ORDER)} gelen={sorted(row.features)}"
            )
        if any(not math.isfinite(float(row.features[key])) for key in LATE_FEATURE_ORDER):
            raise ValueError(f"{row.question_id}: özellik değerleri sonlu olmalı")


def _feature_matrix(rows: Sequence[LateCalibrationRow]) -> np.ndarray:
    return np.array(
        [[float(row.features[name]) for name in LATE_FEATURE_ORDER] for row in rows],
        dtype=np.float64,
    )


def late_calibration_key(identity: Mapping[str, Any]) -> str:
    """Ana indeks + BM25 + iki ColBERT kimliğinin içerik-tabanlı anahtarı."""
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"late-channel-v1__{digest}"


@dataclass(frozen=True)
class LateCalibrationArtifact:
    """Dört geç-kanal özelliği için model, eşik ve yeniden oynatma künyesi."""

    key: str
    identity: dict[str, Any]
    calibrator: Calibrator
    threshold: dict[str, Any]
    kunye: dict[str, Any]
    schema_version: int = LATE_ARTIFACT_SCHEMA_VERSION

    @property
    def tau(self) -> float:
        return float(self.threshold["value"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key": self.key,
            "identity": self.identity,
            "calibrator": self.calibrator.to_dict(),
            "threshold": self.threshold,
            "kunye": self.kunye,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LateCalibrationArtifact:
        return cls(
            schema_version=int(data["schema_version"]),
            key=str(data["key"]),
            identity=dict(data["identity"]),
            calibrator=Calibrator.from_dict(data["calibrator"]),
            threshold=dict(data["threshold"]),
            kunye=dict(data["kunye"]),
        )

    def save(self, directory: Path | str) -> Path:
        """Artefaktı atomik yaz; yarım JSON hiçbir zaman geçerli sonuç olmaz."""
        path = Path(directory)
        if path.suffix == ".json":
            target = path
        else:
            target = path / LATE_ARTIFACT_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(self.to_dict(), ensure_ascii=False, indent=1, sort_keys=True) + "\n"
        tmp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_name = handle.name
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        finally:
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: Path | str, *, expected_key: str) -> LateCalibrationArtifact:
        source = Path(path)
        if source.is_dir():
            source = source / LATE_ARTIFACT_FILENAME
        if not source.exists():
            raise FileNotFoundError(f"geç kanal kalibrasyon artefaktı yok: {source}")
        artifact = cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
        if artifact.schema_version != LATE_ARTIFACT_SCHEMA_VERSION:
            raise LateCalibrationMismatch(
                "geç kanal artefakt şeması uyuşmuyor: "
                f"{artifact.schema_version} != {LATE_ARTIFACT_SCHEMA_VERSION}"
            )
        if artifact.key != expected_key:
            raise LateCalibrationMismatch(
                "geç kanal kalibrasyon anahtarı uyuşmuyor: "
                f"artefakt={artifact.key!r} beklenen={expected_key!r}"
            )
        if tuple(artifact.calibrator.feature_names) != LATE_FEATURE_ORDER:
            raise LateCalibrationMismatch(
                "geç kanal özellik sırası uyuşmuyor: "
                f"{artifact.calibrator.feature_names!r} != {LATE_FEATURE_ORDER!r}"
            )
        if late_calibration_key(artifact.identity) != artifact.key:
            raise LateCalibrationMismatch("artefakt kimliği kendi anahtarıyla uyuşmuyor")
        return artifact


def _counts(rows: Sequence[LateCalibrationRow]) -> dict[str, int]:
    return {
        "total": len(rows),
        "positive": sum(row.label for row in rows),
        "negative": sum(1 for row in rows if row.label == 0),
    }


def evaluate_late_calibration(
    artifact: LateCalibrationArtifact, rows: Sequence[LateCalibrationRow]
) -> dict[str, Any]:
    """Sabit model ve tau'yu verilen yakada değerlendir; asla yeniden fit etme."""
    validate_partition(rows, name="evaluation")
    X = _feature_matrix(rows)
    labels = np.array([row.label for row in rows], dtype=np.float64)
    probs = artifact.calibrator.predict_proba(X)
    tau = artifact.tau
    answered = probs >= tau
    coverage = float(np.mean(answered))
    risk = float(np.mean(labels[answered] == 0.0)) if bool(np.any(answered)) else None
    curve = risk_coverage(probs, labels)

    safe = labels == 1.0
    n_safe = int(np.sum(safe))
    safe_accepted = int(np.sum(answered & safe))
    answerable = np.array([row.answerable for row in rows], dtype=bool)
    unanswerable_rate, n_unanswerable, n_unanswerable_errors, unanswerable_upper = (
        false_answer_rate_on_unanswerable(probs, answerable, tau)
    )

    return {
        "counts": _counts(rows),
        "tau": tau,
        "brier": brier(probs, labels),
        "ece": ece(probs, labels),
        "auroc": auroc(probs, labels),
        "aurc": risk_coverage_auc(curve),
        "coverage_at_tau": coverage,
        "risk_at_tau": risk,
        "risk_coverage": [
            {"tau": float(point_tau), "coverage": cov, "risk": point_risk}
            for point_tau, cov, point_risk in curve
        ],
        "safe_answerable_accept": {
            "rate": safe_accepted / n_safe,
            "n": n_safe,
            "accepted": safe_accepted,
        },
        "false_answer_on_unanswerable": {
            "rate": unanswerable_rate,
            "n": n_unanswerable,
            "errors": n_unanswerable_errors,
            "upper_bound_95": unanswerable_upper,
            "method": "clopper_pearson",
        },
        "per_question": [
            {
                "qid": row.question_id,
                "outer_split": row.outer_split,
                "inner_split": row.inner_split,
                "group": row.group,
                "answerable": row.answerable,
                "label": row.label,
                "gold_in_topk": row.gold_in_topk,
                "unanswerable_reason": row.unanswerable_reason,
                "slice": row.slice,
                "source": row.source,
                "prob": float(prob),
                "answered_at_tau": bool(is_answered),
                "features": {key: float(value) for key, value in row.features.items()},
                "diagnostics": {key: float(value) for key, value in row.diagnostics.items()},
            }
            for row, prob, is_answered in zip(rows, probs.tolist(), answered.tolist(), strict=True)
        ],
    }


def fit_late_calibration(
    fit_rows: Sequence[LateCalibrationRow],
    calibration_rows: Sequence[LateCalibrationRow],
    *,
    identity: Mapping[str, Any],
    max_risk: float = DEFAULT_MAX_RISK,
    data_kunye: Mapping[str, Any] | None = None,
) -> LateCalibrationArtifact:
    """Fit'i ayrı yakada yap, tau'yu yalnız calibration yakasında seç."""
    validate_partition(fit_rows, name="fit")
    validate_partition(calibration_rows, name="calibration")
    calibrator = Calibrator.fit(
        _feature_matrix(fit_rows),
        np.array([row.label for row in fit_rows], dtype=np.float64),
        feature_names=LATE_FEATURE_ORDER,
    )
    calibration_probs = calibrator.predict_proba(_feature_matrix(calibration_rows))
    calibration_labels = np.array([row.label for row in calibration_rows], dtype=np.float64)
    choice = choose_threshold(calibration_probs, calibration_labels, max_risk=max_risk)
    identity_copy = json.loads(json.dumps(identity, ensure_ascii=False, sort_keys=True))
    artifact = LateCalibrationArtifact(
        key=late_calibration_key(identity_copy),
        identity=identity_copy,
        calibrator=calibrator,
        threshold={"max_risk": max_risk, **choice.to_dict()},
        kunye={
            "fit_counts": _counts(fit_rows),
            "calibration_counts": _counts(calibration_rows),
            **(dict(data_kunye) if data_kunye else {}),
        },
    )
    artifact.kunye["fit_metrics"] = evaluate_late_calibration(artifact, fit_rows)
    artifact.kunye["calibration_metrics"] = evaluate_late_calibration(artifact, calibration_rows)
    return artifact


def enablement_verdict(metrics: Mapping[str, Any], *, identity_matches: bool) -> dict[str, Any]:
    """Kilitli test metriklerini ürün bayrağının beş koşuluna çevirir."""
    risk = metrics.get("risk_at_tau")
    safe_rate = float(metrics["safe_answerable_accept"]["rate"])
    unanswerable = metrics["false_answer_on_unanswerable"]
    checks = {
        "selective_risk_point_lte_0_05": risk is not None and float(risk) <= 0.05,
        "unanswerable_false_accept_point_lte_0_02": float(unanswerable["rate"]) <= 0.02,
        "unanswerable_false_accept_cp95_lte_0_05": float(unanswerable["upper_bound_95"]) <= 0.05,
        "safe_answerable_accept_rate_gte_0_80": safe_rate >= 0.80,
        "identity_matches": bool(identity_matches),
    }
    return {
        "eligible_to_enable": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_selective_risk": 0.05,
            "max_unanswerable_false_accept": 0.02,
            "max_unanswerable_cp95": 0.05,
            "min_safe_answerable_accept": 0.80,
        },
    }
