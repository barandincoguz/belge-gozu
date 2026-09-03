import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, get_args

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
# Cevaplanamazlık gerekçesi sözlüğü. `eksik-kanit` sınırdaki sınıftır: soru
# KORPUSTAKİ bir kanun hakkındadır ama aranan somut ayrıntı (yönetmeliğe
# devredilmiş bir usul, kanunun yazmadığı bir tutar, mülga bir hüküm) korpus
# metninde YOKTUR. `korpus-disi`den farkı, konunun korpusta bulunmasıdır; bu
# yüzden retrieval ilgili belgeyi getirir ve model "kanıt var" sanabilir —
# kalibrasyonun en zor durumu. `abstention_eval_v1.jsonl` bu sınıfı doldurur.
AbstentionEvalReason = Literal["korpus-disi", "eksik-kanit", "anlamsiz", "belirsiz"]

# `ajan-taslak`: model ajanının yazdığı, İNSAN ONAYI ALMAMIŞ satır. Mevcut
# `ajan-taslak-insan-onayli` ile karıştırılmasın diye ayrı bir değerdir —
# aksi halde onaysız satırlar künyede onaylı gibi sayılırdı.
SourceType = Literal["insan", "insan-paraphrase", "ajan-taslak-insan-onayli", "ajan-taslak"]

_SLICES: tuple[Slice, ...] = get_args(Slice)


class VerificationLevel(StrEnum):
    mechanical = "mechanical"
    model_cross_check = "model-cross-check"
    human = "human"


VERIFICATION_RANK = {
    VerificationLevel.mechanical: 0,
    VerificationLevel.model_cross_check: 1,
    VerificationLevel.human: 2,
}


def verification_level(kind: str) -> VerificationLevel:
    """Collapse concrete verification kinds onto the public strength scale."""
    if kind == "mechanical:manifest-absence":
        return VerificationLevel.mechanical
    return VerificationLevel(kind)


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
    source_type: SourceType
    requires_visual: bool
    requires_multi_hop: bool
    unanswerable_reason: AbstentionEvalReason | None
    verified_by: str
    verification_status: Literal["draft", "verified", "rejected"]
    # insan doğrulama notu (ör. "h yanlış sayfa"); geriye dönük uyumlu (varsayılan "")
    # — scripts/verify_retrieval_eval.py --review tarafından yazılır.
    verification_note: str = ""
    # Doğrulamayı KİMİN yaptığı değil, NE TÜR bir doğrulama olduğu: insan
    # doğrulaması ile model çapraz-doğrulaması ayrı şeylerdir ve kapı
    # raporlarında ayrı sayılır. Model çapraz-kontrolü sayfa görüntülerini
    # yeniden okuyan bağımsız bir model turudur — insan onayı YERİNE GEÇMEZ;
    # bu alan olmadan iki tür `verified` satır birbirine karışır ve birleşik
    # sayı yanlışlıkla "insan doğrulanmış" diye okunabilir.
    # Varsayılan "human": alan eklenmeden önce yazılmış satırlar geçerli kalır.
    #
    # "mechanical:manifest-absence" ÜÇÜNCÜ ve en zayıf türdür: hiç kimse (insan
    # ya da model) sorunun cevabını aramamıştır; yalnızca bir betik, sorunun
    # dayandığı kanunun korpus manifestinde BULUNMADIĞINI göstermiştir
    # (scripts/validate_abstention_eval.py). Bu, "cevaplanamaz" iddiasını kanıtlamaz —
    # yalnız "dayanak belge korpusta yok" iddiasını kanıtlar; artık risk,
    # korpustaki BAŞKA bir kanunun aynı soruyu cevaplayabilmesidir. Bu yüzden
    # ayrı bir değer: `verified` sayısı türlere ayrılmadan okunursa mekanik
    # satırlar insan onayı sanılır.
    verification_kind: Literal["human", "model-cross-check", "mechanical:manifest-absence"] = (
        "human"
    )

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


class BenchSelection(BaseModel):
    questions: list[BenchQuestion]
    total: int
    selected: int
    filtered_out: int
    only_verified: bool
    min_verification: VerificationLevel | None

    def provenance(self) -> dict[str, bool | int | str | None]:
        return {
            "only_verified": self.only_verified,
            "min_verification": self.min_verification.value if self.min_verification else None,
            "total": self.total,
            "selected": self.selected,
            "filtered_out": self.filtered_out,
        }


def select_bench(
    path: Path | str,
    only_verified: bool = True,
    min_verification: VerificationLevel | str | None = None,
) -> BenchSelection:
    """Load a JSONL benchmark and report exactly what selection removed."""
    path = Path(path)
    all_questions: list[BenchQuestion] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            q = BenchQuestion(**rec)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as e:
            raise ValueError(f"bench satır {i}: {e}") from e
        all_questions.append(q)

    minimum = VerificationLevel(min_verification) if min_verification is not None else None
    questions: list[BenchQuestion] = []
    for q in all_questions:
        if (only_verified or minimum is not None) and q.verification_status != "verified":
            continue
        if minimum is not None:
            actual = verification_level(q.verification_kind)
            if VERIFICATION_RANK[actual] < VERIFICATION_RANK[minimum]:
                continue
        questions.append(q)
    if not questions:
        raise ValueError("bench boş: yüklenecek soru yok")
    return BenchSelection(
        questions=questions,
        total=len(all_questions),
        selected=len(questions),
        filtered_out=len(all_questions) - len(questions),
        only_verified=only_verified,
        min_verification=minimum,
    )


def load_bench(
    path: Path | str,
    only_verified: bool = True,
    min_verification: VerificationLevel | str | None = None,
) -> list[BenchQuestion]:
    """Compatibility wrapper returning only selected benchmark questions."""
    return select_bench(
        path,
        only_verified=only_verified,
        min_verification=min_verification,
    ).questions


def bench_stats(questions: list[BenchQuestion]) -> dict[str, int]:
    """Dilim (slice) başına soru sayımı; tüm dilimler 0 varsayılanıyla dahil."""
    stats: dict[str, int] = {s: 0 for s in _SLICES}
    for q in questions:
        stats[q.slice] += 1
    return stats


def load_splits(path: Path | str) -> dict[str, set[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
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


def _hash50(key: str) -> Literal["dev", "test"]:
    """sha256 tabanlı kararlı 50/50 atama (tohum yok — anahtarın kendisi tohumdur)."""
    return "dev" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def _field(question: "Mapping[str, Any] | BenchQuestion", name: str) -> Any:
    if isinstance(question, BenchQuestion):
        return getattr(question, name, None)
    return question.get(name)


def assign_split(
    question: "Mapping[str, Any] | BenchQuestion",
    splits: dict[str, set[str]],
) -> Literal["dev", "test"]:
    """Bir soruyu dev/test'e atar — SAF fonksiyon, dosya/rastgelelik yok.

    Kural (aynısı `data/bench/splits_v1.json` içinde de yazılıdır):

    1. Cevaplanabilir soru (`gold_doc_ids` dolu) → `gold_doc_ids[0]` hangi
       kümedeyse orası; hiçbirinde değilse `dev` (güvenli varsayılan).
    2. `korpus-disi` + `_anchor_law` var → `sha256("anchor:<kanun no>")` ile
       50/50. Kanun bazında gruplanır: aynı absent kanuna dayanan tüm sorular
       aynı yakaya düşer, böylece test kümesi dev'de görülmüş bir kanunu
       tekrar sormaz.
    3. `eksik-kanit` + `_subject_doc` var → konu belgesi `test_docs` içindeyse
       `test`, değilse `dev`. Bu sınıf korpustaki bir belgeye bağlı olduğu için
       cevaplanabilir sorularla AYNI hukuk-gruplu bölmeyi paylaşmalıdır.
    4. Diğer (anlamsız-ood; ya da eksik alan) → `sha256("qid:<id>")` ile 50/50.

    `_anchor_law` / `_subject_doc` alt çizgili alanlardır: `BenchQuestion`
    onları taşımaz (pydantic yok sayar), bu yüzden fonksiyon ham JSONL
    sözlüğünü de kabul eder. `BenchQuestion` verilirse 2-3 numaralı kurallar
    veri olmadığından uygulanamaz ve 4'e düşülür — çağıran taraf hukuk-gruplu
    atama istiyorsa ham satırı geçmelidir.
    """
    gold_doc_ids = _field(question, "gold_doc_ids") or []
    if gold_doc_ids:
        primary = gold_doc_ids[0]
        if primary in splits.get("test_docs", set()):
            return "test"
        return "dev"

    reason = _field(question, "unanswerable_reason")
    qid = str(_field(question, "question_id") or "")

    if reason == "korpus-disi":
        anchor = _field(question, "_anchor_law")
        if anchor:
            return _hash50(f"anchor:{anchor}")
    elif reason == "eksik-kanit":
        subject = _field(question, "_subject_doc")
        if subject:
            return "test" if subject in splits.get("test_docs", set()) else "dev"

    return _hash50(f"qid:{qid}")
