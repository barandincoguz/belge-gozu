import hashlib
import json
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ValidationError, model_validator

QueryStyle = Literal["dogal", "hukuki", "madde-referansli", "anahtar-kelime"]
Slice = Literal[
    "dogrudan-madde",
    "paraphrase",
    "madde-numarali",
    "ayni-kanun-hard-negative",
    "capraz-kanun-terim",
    "tablo-layout",
    "tarihi-tarama",
    "belirsiz-coklu-dayanak",
    "multi-hop",
    "korpus-disi",
    "eksik-kanit",
    "anlamsiz-ood",
]
UnansReason = Literal["korpus-disi", "eksik-kanit", "anlamsiz", "belirsiz"]

_SLICES: tuple[Slice, ...] = get_args(Slice)


class BenchQuestion(BaseModel):
    question_id: str
    question: str
    query_style: QueryStyle
    answerable: bool
    gold_doc_ids: list[str]
    gold_page_ids: list[str]  # "dok:sayfa"; answerable=True iken >=1
    gold_article_ids: list[str]  # "k4721:m19" / "k4721:gm2"; boş olabilir
    minimal_evidence_spans: list[str]
    reference_answer: str  # answerable=False iken ""
    slice: Slice
    difficulty: Literal["kolay", "orta", "zor"]
    source_type: Literal["insan", "insan-paraphrase", "ajan-taslak-insan-onayli"]
    requires_visual: bool
    requires_multi_hop: bool
    unanswerable_reason: UnansReason | None
    verified_by: str
    verification_status: Literal["draft", "verified", "rejected"]
    # insan doğrulama notu (ör. "h yanlış sayfa"); geriye dönük uyumlu (varsayılan "")
    # — scripts/verify_canary.py --review tarafından yazılır.
    verification_note: str = ""

    @model_validator(mode="after")
    def _check_answerability(self) -> "BenchQuestion":
        if self.answerable:
            if not self.gold_page_ids:
                raise ValueError("answerable=True iken gold_page_ids boş olamaz")
            if not self.reference_answer:
                raise ValueError("answerable=True iken reference_answer boş olamaz")
            if self.unanswerable_reason is not None:
                raise ValueError("answerable=True iken unanswerable_reason None olmalı")
        else:
            if self.gold_page_ids != []:
                raise ValueError("answerable=False iken gold_page_ids boş olmalı")
            if self.unanswerable_reason is None:
                raise ValueError("answerable=False iken unanswerable_reason zorunlu")
        return self

    @model_validator(mode="after")
    def _check_gold_page_doc_consistency(self) -> "BenchQuestion":
        for gp in self.gold_page_ids:
            if ":" not in gp:
                raise ValueError(f"gold_page_ids elemanı ':' içermeli: {gp!r}")
            doc_id = gp.split(":", 1)[0]
            if doc_id not in self.gold_doc_ids:
                raise ValueError(f"gold_page_ids doc'u gold_doc_ids'te yok: {gp!r}")
        return self

    @model_validator(mode="after")
    def _check_verification(self) -> "BenchQuestion":
        if self.verification_status == "verified" and not self.verified_by:
            raise ValueError("verification_status='verified' iken verified_by boş olamaz")
        return self


def load_bench(path: Path, only_verified: bool = True) -> list[BenchQuestion]:
    """JSONL bench dosyasını satır satır okur (satır no. 1'den başlar)."""
    questions: list[BenchQuestion] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            q = BenchQuestion(**rec)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as e:
            raise ValueError(f"bench satır {i}: {e}") from e
        if only_verified and q.verification_status != "verified":
            continue
        questions.append(q)
    if not questions:
        raise ValueError("bench boş: yüklenecek soru yok")
    return questions


def bench_stats(questions: list[BenchQuestion]) -> dict[str, int]:
    """Dilim (slice) başına soru sayımı; tüm dilimler 0 varsayılanıyla dahil."""
    stats: dict[str, int] = {s: 0 for s in _SLICES}
    for q in questions:
        stats[q.slice] += 1
    return stats


def load_splits(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dev_docs": set(data.get("dev_docs", [])),
        "test_docs": set(data.get("test_docs", [])),
    }


def question_split(q: BenchQuestion, splits: dict[str, set[str]]) -> Literal["dev", "test"]:
    if q.gold_doc_ids:
        primary_doc = q.gold_doc_ids[0]
        if primary_doc in splits["dev_docs"]:
            return "dev"
        if primary_doc in splits["test_docs"]:
            return "test"
        # T12 öncesi doldurulmamış split → güvenli varsayılan dev (hiçbir kümede yok)
        return "dev"
    digest = hashlib.sha256(q.question_id.encode()).hexdigest()
    return "dev" if int(digest, 16) % 2 == 0 else "test"
