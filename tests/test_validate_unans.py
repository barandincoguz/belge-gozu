"""`scripts/validate_unans.py` saf mantığı — dosya/parquet I/O olmadan.

Doğrulayıcının değeri, hatalı bir satırı GERÇEKTEN yakalamasındadır; bu yüzden
testler yalnız temiz girdiyi değil, her kontrol için kasıtlı bozuk bir satırı da
geçirir. Aksi halde "TEMİZ" çıktısı hiçbir şey kanıtlamaz.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_unans", REPO / "scripts" / "validate_unans.py"
)
assert _spec and _spec.loader
vu = importlib.util.module_from_spec(_spec)
sys.modules["validate_unans"] = vu
_spec.loader.exec_module(vu)


CORPUS_IDS = {"k4857", "k6098", "k1512", "rg1935a", "rg1945a"}
CORPUS_NAMES = {
    "k4857": "İş Kanunu",
    "k6098": "Türk Borçlar Kanunu",
    "k1512": "Noterlik Kanunu",
    "rg1935a": "RG arşiv örneği 2",
    "rg1945a": "RG arşiv örneği 3",
}


def base_row(**over) -> dict:
    row = {
        "question_id": "u001",
        "question": "5901 sayılı Türk Vatandaşlığı Kanunu'na göre ikamet şartı nedir?",
        "query_style": "hukuki",
        "answerable": False,
        "gold_doc_ids": [],
        "gold_page_ids": [],
        "gold_article_ids": [],
        "minimal_evidence_spans": [],
        "reference_answer": "",
        "slice": "korpus-disi",
        "difficulty": "orta",
        "source_type": "ajan-taslak",
        "requires_visual": False,
        "requires_multi_hop": False,
        "unanswerable_reason": "korpus-disi",
        "verified_by": "script:validate_unans",
        "verification_status": "verified",
        "verification_note": "not",
        "verification_kind": "mechanical:manifest-absence",
        "_anchor_law": "5901",
        "_anchor_name": "Türk Vatandaşlığı Kanunu",
    }
    row.update(over)
    return row


def errors_for(row: dict) -> list[str]:
    """Tek satırı sınar.

    Küme çapındaki iki kontrol elenir — tek satırlık girdiyle sağlanamazlar ve
    kendi testleri var: dilim sayımı (200/60/40) ve id'nin satır sırasına
    eşitliği (`u{i:03d}`).
    """
    errs = vu.check_rows([row], CORPUS_IDS, CORPUS_NAMES)
    return [e for e in errs if not e.startswith("dilim ") and "beklenen id" not in e]


def test_id_must_match_row_order():
    """u001 birinci satırda olmalı — kayan numaralandırma sessizce geçmemeli."""
    errs = vu.check_rows([base_row(question_id="u007")], CORPUS_IDS, CORPUS_NAMES)
    assert any("beklenen id u001, bulunan u007" in e for e in errs)


def test_slice_counts_must_match_expected():
    errs = vu.check_rows([base_row()], CORPUS_IDS, CORPUS_NAMES)
    assert any(e.startswith("dilim korpus-disi: 1 satır") for e in errs)


# --------------------------------------------------------------- normalizasyon
def test_tr_lower_handles_dotted_capital_i():
    """Python'ın .lower()'ı yerel-ayara duyarsız: 'İ' -> 'i̇' (birleşik nokta)."""
    assert vu.tr_lower("İŞ KANUNU") == "iş kanunu"
    assert vu.tr_lower("IŞIK") == "ışık"


def test_name_tokens_drops_boilerplate():
    """'sayılı/hakkında/kanunu' gibi kelimeler ayırt edici değildir."""
    assert vu.name_tokens("Afet Sigortaları Kanunu") == {"afet", "sigortalari"}
    assert "kanunu" not in vu.name_tokens("Tapu Kanunu")


def test_jaccard_bounds():
    assert vu.jaccard({"a"}, {"a"}) == 1.0
    assert vu.jaccard({"a"}, {"b"}) == 0.0
    assert vu.jaccard(set(), {"a"}) == 0.0


def test_corpus_law_numbers_skips_rg_scans():
    assert vu.corpus_law_numbers(CORPUS_IDS) == {"4857", "6098", "1512"}


# ----------------------------------------------------------- satır kontrolleri
def test_clean_row_passes():
    assert errors_for(base_row()) == []


def test_anchor_present_in_corpus_is_rejected():
    """k1512 KORPUSTA — 1512'ye çapalamak sessizce geçmemeli."""
    bad = base_row(
        _anchor_law="1512",
        _anchor_name="Noterlik Kanunu",
        question="1512 sayılı Noterlik Kanunu'na göre ücret nedir?",
    )
    errs = errors_for(bad)
    assert any("ÇAPA KORPUSTA" in e for e in errs)


def test_anchor_name_matching_corpus_doc_is_rejected():
    """Numara farklı olsa bile ad korpus belgesiyle örtüşüyorsa yakalanmalı."""
    bad = base_row(
        _anchor_law="9999",
        _anchor_name="Türk Borçlar Kanunu",
        question="9999 sayılı Türk Borçlar Kanunu'na göre faiz nedir?",
    )
    errs = errors_for(bad)
    assert any("çapa adı korpus belgesiyle örtüşüyor" in e for e in errs)


def test_question_must_mention_its_anchor():
    bad = base_row(question="İkamet şartı kaç yıldır?")
    assert any("çapayı anmıyor" in e for e in errors_for(bad))


def test_question_may_name_anchor_without_number():
    """Numara yoksa adın ayırt edici tokenları yeterlidir (doğal dil sorular)."""
    ok = base_row(question="Türk Vatandaşlığı Kanunu'na göre ikamet şartı kaç yıldır?")
    assert errors_for(ok) == []


def test_eksik_kanit_subject_doc_must_be_in_corpus():
    bad = base_row(
        question_id="u261",
        slice="eksik-kanit",
        unanswerable_reason="eksik-kanit",
        verified_by="",
        verification_status="draft",
        verification_kind="model-cross-check",
        _subject_doc="k9999",
    )
    bad.pop("_anchor_law")
    bad.pop("_anchor_name")
    assert any("_subject_doc korpusta yok" in e for e in errors_for(bad))


def test_eksik_kanit_with_corpus_subject_passes():
    ok = base_row(
        question_id="u261",
        slice="eksik-kanit",
        unanswerable_reason="eksik-kanit",
        verified_by="",
        verification_status="draft",
        verification_kind="model-cross-check",
        _subject_doc="k4857",
    )
    ok.pop("_anchor_law")
    ok.pop("_anchor_name")
    assert errors_for(ok) == []


def test_answerable_row_is_rejected():
    bad = base_row(
        answerable=True,
        gold_doc_ids=["k4857"],
        gold_page_ids=["k4857:1"],
        reference_answer="x",
        unanswerable_reason=None,
    )
    errs = errors_for(bad)
    assert any("answerable False olmalı" in e for e in errs)


def test_human_source_type_is_rejected():
    """Bu set insan onaylı değil; `insan` etiketi künyeyi yanlışlar."""
    assert any("ajan-taslak" in e for e in errors_for(base_row(source_type="insan")))


def test_anlamsiz_slice_must_not_be_self_verified():
    """Ayrı denetleyici bekleyen dilim `verified` görünemez."""
    bad = base_row(
        question_id="u201",
        slice="anlamsiz-ood",
        unanswerable_reason="anlamsiz",
        verification_status="verified",
        verified_by="script:validate_unans",
        verification_kind="mechanical:manifest-absence",
    )
    bad.pop("_anchor_law")
    bad.pop("_anchor_name")
    assert any("doğrulama künyesi" in e for e in errors_for(bad))


def test_bad_question_id_format_and_order():
    assert any("biçimi bozuk" in e for e in errors_for(base_row(question_id="x1")))


def test_duplicate_question_ids_are_caught():
    errs = vu.check_rows([base_row(), base_row()], CORPUS_IDS, CORPUS_NAMES)
    assert any("tekrar ediyor" in e for e in errs)


def test_empty_verification_note_is_rejected():
    assert any("verification_note" in e for e in errors_for(base_row(verification_note="")))


# ------------------------------------------------------------- yakın-tekrarlar
def test_near_dupes_flags_reworded_pair():
    rows = [
        {"question_id": "u001", "question": "Pasaport Kanunu'na göre harç ne kadardır?"},
        {"question_id": "u002", "question": "Pasaport Kanunu'na göre ne kadardır harç?"},
    ]
    hits = vu.find_near_dupes(rows, [])
    assert ("u001", "u002", 1.0) in hits


def test_near_dupes_ignores_distinct_questions():
    rows = [
        {"question_id": "u001", "question": "Orman Kanunu'na göre 2/B arazisi nedir?"},
        {"question_id": "u002", "question": "Maden Kanunu'na göre devlet hakkı nasıl hesaplanır?"},
    ]
    assert vu.find_near_dupes(rows, []) == []


def test_near_dupes_checks_against_canary():
    rows = [{"question_id": "u001", "question": "Yerleşim yeri nedir?"}]
    canary = [{"question_id": "c002", "question": "Yerleşim yeri nedir?"}]
    hits = vu.find_near_dupes(rows, canary)
    assert hits and hits[0][1] == "canary:c002"


# -------------------------------------------------------------- split türetimi
DOC_TYPES = {
    "k4857": "kanun",
    "k6098": "kanun",
    "k1512": "kanun",
    "rg1935a": "rg_tarihi",
    "rg1945a": "rg_tarihi",
}


def test_derive_test_docs_is_deterministic_and_sized():
    kw = dict(
        corpus_ids=CORPUS_IDS,
        doc_types=DOC_TYPES,
        seed="s",
        pinned=["k6098"],
        size=3,
        canary_docs={"k6098"},
    )
    first = vu.derive_test_docs(**kw)
    assert first == vu.derive_test_docs(**kw)
    assert len(first) == 3
    assert "k6098" in first


def test_derive_test_docs_guarantees_two_rg_docs():
    out = vu.derive_test_docs(CORPUS_IDS, DOC_TYPES, "s", ["k6098"], 3, {"k6098"})
    assert sum(1 for d in out if DOC_TYPES[d] == "rg_tarihi") >= 2


def test_derive_test_docs_excludes_unpinned_canary_docs():
    """Sabitlenmemiş canary belgesi doldurmaya kapalıdır (26/17 hedefi korunur)."""
    out = vu.derive_test_docs(CORPUS_IDS, DOC_TYPES, "s", ["k6098"], 3, {"k6098", "k4857"})
    assert "k4857" not in out


def test_seed_change_alters_selection():
    a = vu.derive_test_docs(CORPUS_IDS, DOC_TYPES, "seed-a", ["k6098"], 4, {"k6098"})
    b = vu.derive_test_docs(CORPUS_IDS, DOC_TYPES, "seed-b", ["k6098"], 4, {"k6098"})
    assert set(a) | set(b) >= {"k6098"}  # sabitli her zaman içeride
    assert isinstance(a, list) and isinstance(b, list)


# ------------------------------------------------------------- korpus tutarlılığı
def test_load_corpus_matches_repo_state_and_manifest():
    ids, names = vu.load_corpus(REPO)
    assert len(ids) == 56
    assert set(names) == ids
    assert "k5490" in ids, "k5490 (Nüfus Hizmetleri) KORPUSTADIR — çapa olarak kullanılamaz"


@pytest.mark.parametrize("doc_id", ["k1512", "k5490", "k4857", "k6098"])
def test_known_present_laws_cannot_be_anchors(doc_id: str):
    """Ezber koruması: korpusta olduğu bilinen kanunlara çapalamak hata vermeli."""
    ids, names = vu.load_corpus(REPO)
    number = doc_id[1:]
    bad = base_row(
        _anchor_law=number,
        _anchor_name=names[doc_id],
        question=f"{number} sayılı {names[doc_id]}'na göre bir soru?",
    )
    errs = [e for e in vu.check_rows([bad], ids, names) if not e.startswith("dilim ")]
    assert any("ÇAPA KORPUSTA" in e for e in errs)
