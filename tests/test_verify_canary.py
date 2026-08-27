import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_canary import (  # noqa: E402
    apply_decision,
    compute_status,
    doc_prefix_consistent,
    gold_image_paths,
    load_raw_rows,
    normalize_for_match,
    normalize_ws,
    parse_decision,
    precheck_question,
    run_precheck,
    span_found,
    tr_lower,
    write_jsonl_atomic,
)

from belge_gozu.bench.dataset import BenchQuestion  # noqa: E402


def q_dict(**over) -> dict:
    base = dict(
        question_id="c001",
        question="Yerleşim yeri nedir?",
        query_style="dogal",
        answerable=True,
        gold_doc_ids=["k4721"],
        gold_page_ids=["k4721:4"],
        gold_article_ids=["k4721:m19"],
        minimal_evidence_spans=[
            "Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
        ],
        reference_answer="Sürekli kalma niyetiyle oturulan yerdir (TMK m.19).",
        slice="paraphrase",
        difficulty="orta",
        source_type="insan",
        requires_visual=False,
        requires_multi_hop=False,
        unanswerable_reason=None,
        verified_by="",
        verification_status="draft",
    )
    base.update(over)
    return base


def bq(**over) -> BenchQuestion:
    return BenchQuestion(**q_dict(**over))


# --------------------------------------------------------------------------
# normalizasyon
# --------------------------------------------------------------------------


def test_normalize_ws_collapses_whitespace_and_newlines():
    assert normalize_ws("  a\n\nb   c\t d ") == "a b c d"


def test_tr_lower_handles_turkish_dotted_and_dotless_i():
    # İ (büyük noktalı) -> i ; I (büyük noktasız/ASCII) -> ı
    assert tr_lower("İKAMETGAH") == "ikametgah"
    assert tr_lower("IŞIK") == "ışık"


def test_normalize_for_match_combines_both():
    a = "Yerleşim  Yeri\nBİR Kimsenin"
    b = "yerleşim yeri bir kimsenin"
    assert normalize_for_match(a) == normalize_for_match(b)


# --------------------------------------------------------------------------
# alıntı (span) eşleşmesi
# --------------------------------------------------------------------------


def test_span_found_exact_match():
    page = "Madde 19- Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
    assert span_found("Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir.", page)


def test_span_found_case_and_whitespace_insensitive():
    page = "...\nYERLEŞİM\nYERİ bir kimsenin sürekli\nkalma niyetiyle oturduğu yerdir.\n..."
    assert span_found("yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir.", page)


def test_span_not_found_when_text_differs():
    page = "Madde 20- Bir kimsenin aynı zamanda birden çok yerleşim yeri olamaz."
    assert not span_found("sürekli kalma niyetiyle oturduğu yerdir", page)


def test_span_found_empty_span_is_trivially_true():
    assert span_found("", "herhangi bir sayfa metni")


# --------------------------------------------------------------------------
# doc-prefix tutarlılığı
# --------------------------------------------------------------------------


def test_doc_prefix_consistent_true():
    assert doc_prefix_consistent(["k4721"], ["k4721:4", "k4721:5"])


def test_doc_prefix_consistent_false_when_prefix_missing():
    assert not doc_prefix_consistent(["k4721"], ["k9999:1"])


def test_doc_prefix_consistent_empty_lists():
    assert doc_prefix_consistent([], [])


# --------------------------------------------------------------------------
# precheck sınıflandırması (TEMİZ / ŞÜPHELİ / MANUEL)
# --------------------------------------------------------------------------


def test_precheck_clean_when_all_spans_match_and_page_known():
    q = bq()
    page_texts = {
        "k4721:4": "Madde 19- Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
    }
    pc = precheck_question(q, page_texts, known_page_ids={"k4721:4"})
    assert pc.group == "TEMİZ"
    assert pc.notes == []
    assert pc.span_checks[0].found is True


def test_precheck_suspicious_when_span_missing():
    q = bq()
    page_texts = {
        "k4721:4": "Madde 20- Bambaşka bir madde metni burada, aranan ifade bu sayfada yok."
    }
    pc = precheck_question(q, page_texts, known_page_ids={"k4721:4"})
    assert pc.group == "ŞÜPHELİ"
    assert pc.span_checks[0].found is False
    assert pc.span_checks[0].closest is not None


def test_precheck_suspicious_when_page_missing_from_index():
    q = bq()
    page_texts = {
        "k4721:4": "Madde 19- Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
    }
    pc = precheck_question(q, page_texts, known_page_ids=set())
    assert pc.group == "ŞÜPHELİ"
    assert any("indekste yok" in n for n in pc.notes)


def test_precheck_manual_when_page_text_too_short():
    q = bq(gold_doc_ids=["rg1928a"], gold_page_ids=["rg1928a:1"], gold_article_ids=[])
    pc = precheck_question(q, {"rg1928a:1": "   "}, known_page_ids={"rg1928a:1"})
    assert pc.group == "MANUEL"


def test_precheck_manual_when_page_text_missing_entirely():
    q = bq()
    pc = precheck_question(q, {"k4721:4": None}, known_page_ids={"k4721:4"})
    assert pc.group == "MANUEL"
    assert any("bulunamadı" in n for n in pc.notes)


def test_precheck_unanswerable_question_is_always_clean():
    q = bq(
        answerable=False,
        gold_doc_ids=[],
        gold_page_ids=[],
        gold_article_ids=[],
        minimal_evidence_spans=[],
        reference_answer="",
        slice="korpus-disi",
        unanswerable_reason="korpus-disi",
    )
    pc = precheck_question(q, {}, known_page_ids=set())
    assert pc.group == "TEMİZ"


def test_run_precheck_multiple_questions():
    q1 = bq(question_id="c001")
    q2 = bq(question_id="c002", minimal_evidence_spans=["hiç bulunmayacak ifade"])
    page_texts = {
        "k4721:4": "Madde 19- Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
    }
    results = run_precheck([q1, q2], page_texts, known_page_ids={"k4721:4"})
    groups = {r.question_id: r.group for r in results}
    assert groups == {"c001": "TEMİZ", "c002": "ŞÜPHELİ"}


# --------------------------------------------------------------------------
# karar ayrıştırma + uygulama (interaktif inceleme — saf kısım)
# --------------------------------------------------------------------------


def test_parse_decision_letter_only():
    assert parse_decision("e") == ("e", "")


def test_parse_decision_with_note():
    assert parse_decision("h yanlış sayfa") == ("h", "yanlış sayfa")


def test_parse_decision_case_insensitive_letter():
    assert parse_decision("E") == ("e", "")


def test_parse_decision_invalid_letter_returns_none():
    assert parse_decision("x") is None
    assert parse_decision("") is None
    assert parse_decision("   ") is None


def test_apply_decision_verify_sets_status_and_verified_by():
    raw = q_dict(_hard_negatives=["k4721:m20"])
    out = apply_decision(raw, "e", "", "baran")
    assert out["verification_status"] == "verified"
    assert out["verified_by"] == "baran"
    assert out["_hard_negatives"] == ["k4721:m20"]  # bilinmeyen alan korunmuş


def test_apply_decision_reject_keeps_verified_by():
    raw = q_dict()
    out = apply_decision(raw, "h", "yanlış sayfa", "baran")
    assert out["verification_status"] == "rejected"
    assert out["verified_by"] == "baran"
    assert out["verification_note"] == "yanlış sayfa"


def test_apply_decision_skip_leaves_status_untouched():
    raw = q_dict(verification_status="draft", verified_by="")
    out = apply_decision(raw, "a", "sonra bakılacak", "baran")
    assert out["verification_status"] == "draft"
    assert out["verified_by"] == ""
    assert out["verification_note"] == "sonra bakılacak"


def test_apply_decision_does_not_mutate_input():
    raw = q_dict()
    apply_decision(raw, "e", "", "baran")
    assert raw["verification_status"] == "draft"  # orijinal dict değişmedi


def test_apply_decision_rejects_quit_letter():
    with pytest.raises(ValueError):
        apply_decision(q_dict(), "q", "", "baran")


# --------------------------------------------------------------------------
# atomik round-trip: _hard_negatives ve bilinmeyen alanlar HİÇ kaybolmamalı
# --------------------------------------------------------------------------


def test_write_jsonl_atomic_roundtrip_preserves_unknown_keys(tmp_path: Path):
    rows = [
        q_dict(
            question_id="c001",
            _hard_negatives=["k4721:m20", "k4721:m22"],
            _reviewer_hint="dikkat",
        ),
        q_dict(question_id="c002"),
    ]
    out = tmp_path / "canary.jsonl"
    write_jsonl_atomic(out, rows)

    text_lines = out.read_text(encoding="utf-8").splitlines()
    loaded = [json.loads(line) for line in text_lines if line.strip()]
    assert loaded == rows
    assert loaded[0]["_hard_negatives"] == ["k4721:m20", "k4721:m22"]
    assert loaded[0]["_reviewer_hint"] == "dikkat"


def test_write_jsonl_atomic_no_leftover_temp_files(tmp_path: Path):
    out = tmp_path / "canary.jsonl"
    write_jsonl_atomic(out, [q_dict()])
    leftovers = [p for p in tmp_path.iterdir() if p.name != "canary.jsonl"]
    assert leftovers == []


def test_write_jsonl_atomic_then_load_raw_rows_matches(tmp_path: Path):
    rows = [q_dict(question_id="c001", _hard_negatives=["x"]), q_dict(question_id="c002")]
    out = tmp_path / "canary.jsonl"
    write_jsonl_atomic(out, rows)
    assert load_raw_rows(out) == rows


# --------------------------------------------------------------------------
# görüntü yolu inşası (saf — I/O yok, dosya var mı bakmaz)
# --------------------------------------------------------------------------


def test_gold_image_paths_single_page(tmp_path: Path):
    q = bq()
    paths = gold_image_paths(q, tmp_path / "images")
    assert paths == [tmp_path / "images" / "k4721" / "0004.webp"]


def test_gold_image_paths_multiple_pages(tmp_path: Path):
    q = bq(gold_doc_ids=["k4721"], gold_page_ids=["k4721:4", "k4721:5"])
    paths = gold_image_paths(q, tmp_path / "images")
    assert paths == [
        tmp_path / "images" / "k4721" / "0004.webp",
        tmp_path / "images" / "k4721" / "0005.webp",
    ]


# --------------------------------------------------------------------------
# durum sayımı (--status)
# --------------------------------------------------------------------------


def test_compute_status_counts_by_status():
    qs = [
        bq(question_id="c001", verification_status="verified", verified_by="baran"),
        bq(question_id="c002", verification_status="draft"),
        bq(question_id="c003", verification_status="rejected", verified_by="baran"),
    ]
    status = compute_status(qs)
    assert status["by_status"] == {"verified": 1, "draft": 1, "rejected": 1}


def test_compute_status_answerable_vs_unanswerable_verified():
    answerable_verified = [
        bq(question_id=f"c{i:03d}", verification_status="verified", verified_by="baran")
        for i in range(25)
    ]
    unanswerable_verified = [
        bq(
            question_id=f"u{i:03d}",
            answerable=False,
            gold_doc_ids=[],
            gold_page_ids=[],
            gold_article_ids=[],
            minimal_evidence_spans=[],
            reference_answer="",
            slice="korpus-disi",
            unanswerable_reason="korpus-disi",
            verification_status="verified",
            verified_by="baran",
        )
        for i in range(5)
    ]
    status = compute_status(answerable_verified + unanswerable_verified)
    assert status["verified_answerable"] == 25
    assert status["verified_unanswerable"] == 5
    assert status["target_met"] is True


def test_compute_status_target_not_met_when_below_threshold():
    qs = [bq(question_id="c001", verification_status="verified", verified_by="baran")]
    status = compute_status(qs)
    assert status["verified_answerable"] == 1
    assert status["verified_unanswerable"] == 0
    assert status["target_met"] is False


def test_compute_status_by_slice_includes_all_slices_with_zero_default():
    qs = [bq(question_id="c001")]
    status = compute_status(qs)
    assert status["by_slice_total"]["paraphrase"] == 1
    assert status["by_slice_total"]["multi-hop"] == 0
    assert status["by_slice_verified"]["paraphrase"] == 0  # taslak, henüz doğrulanmamış
