import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_retrieval_eval import (  # noqa: E402
    PARAPHRASE_MAX_OVERLAP,
    apply_decision,
    compute_status,
    content_overlap,
    doc_prefix_consistent,
    gold_image_paths,
    load_raw_rows,
    normalize_for_match,
    normalize_ws,
    parse_decision,
    precheck_question,
    run_precheck,
    select_review_queue,
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
    # c001 gerçek veride `dogrudan-madde`; varsayılan `paraphrase` dilimi
    # örtüşme kapısına takılır (bkz. test_precheck_flags_paraphrase_row_...).
    q = bq(slice="dogrudan-madde")
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
    # c001/c002 gerçek veride `dogrudan-madde` (çapraz-kontrol turu düzeltti);
    # varsayılan `paraphrase` dilimi örtüşme kapısına takılırdı.
    q1 = bq(question_id="c001", slice="dogrudan-madde")
    q2 = bq(
        question_id="c002",
        slice="dogrudan-madde",
        minimal_evidence_spans=["hiç bulunmayacak ifade"],
    )
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
    out = tmp_path / "retrieval_eval.jsonl"
    write_jsonl_atomic(out, rows)

    text_lines = out.read_text(encoding="utf-8").splitlines()
    loaded = [json.loads(line) for line in text_lines if line.strip()]
    assert loaded == rows
    assert loaded[0]["_hard_negatives"] == ["k4721:m20", "k4721:m22"]
    assert loaded[0]["_reviewer_hint"] == "dikkat"


def test_write_jsonl_atomic_no_leftover_temp_files(tmp_path: Path):
    out = tmp_path / "retrieval_eval.jsonl"
    write_jsonl_atomic(out, [q_dict()])
    leftovers = [p for p in tmp_path.iterdir() if p.name != "retrieval_eval.jsonl"]
    assert leftovers == []


def test_write_jsonl_atomic_then_load_raw_rows_matches(tmp_path: Path):
    rows = [q_dict(question_id="c001", _hard_negatives=["x"]), q_dict(question_id="c002")]
    out = tmp_path / "retrieval_eval.jsonl"
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


# --------------------------------------------------------------------------
# paraphrase sözlüksel örtüşme kapısı
#
# Çapraz-kontrol turunun bulduğu 6 kusurun 3'ü (c001, c002, c108) tek bir
# hataydı: soru, gold sayfanın KENDİ sözcüklerini tekrarlıyor ama `paraphrase`
# etiketli. Bu, dilimi ölçtüğü şeyden koparır — "yeniden ifade edilmiş sorgu"
# dilimi aslında birebir kelime eşleşmesini ölçer ve BM25 haksız yere iyi
# görünür. Kapı, etiketi görüşten ÖLÇÜME çevirir.
# --------------------------------------------------------------------------


def test_content_overlap_is_full_when_question_reuses_page_vocabulary():
    """c001'in ta kendisi: soru maddenin terimlerini aynen taşıyor."""
    page = "Madde 19- Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
    assert content_overlap("Yerleşim yeri nedir?", page) == 1.0


def test_content_overlap_is_low_for_genuinely_reworded_question():
    page = "Madde 19- Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
    reworded = "Bir kişinin resmî adresi hangi ölçüte göre saptanır?"
    assert content_overlap(reworded, page) < PARAPHRASE_MAX_OVERLAP


def test_content_overlap_is_spelling_invariant():
    """exp12 yazım-değişmezliği: aksanlı ve aksansız soru aynı oranı vermeli."""
    page = "Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
    assert content_overlap("Yerleşim yeri nedir?", page) == content_overlap(
        "Yerlesim yeri nedir?", page
    )


def test_content_overlap_is_zero_when_question_has_no_content_tokens():
    """Yalnız stopword'den oluşan soru sıfır döner (sıfıra bölme değil)."""
    assert content_overlap("ve veya ile", "herhangi bir sayfa metni") == 0.0


def test_precheck_flags_paraphrase_row_that_reuses_page_vocabulary():
    q = bq(slice="paraphrase")
    page_texts = {
        "k4721:4": "Madde 19- Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
    }
    pc = precheck_question(q, page_texts, known_page_ids={"k4721:4"})
    assert pc.group == "ŞÜPHELİ"
    assert any("paraphrase" in n and "örtüşme" in n for n in pc.notes)


def test_precheck_does_not_apply_overlap_gate_outside_paraphrase_slice():
    """dogrudan-madde diliminde yüksek örtüşme BEKLENEN davranıştır, kusur değil."""
    q = bq(slice="dogrudan-madde")
    page_texts = {
        "k4721:4": "Madde 19- Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
    }
    pc = precheck_question(q, page_texts, known_page_ids={"k4721:4"})
    assert pc.group == "TEMİZ"
    assert pc.notes == []


# --------------------------------------------------------------------------
# insan inceleme kuyruğu (HTML arayüzünün saf mantığı)
#
# v1'in 48 satırının hepsi `verification_status: verified`, ama 45'i
# `model-cross-check`. Terminal `--review` yalnız `draft` gösterdiği için bu 45
# satır hiçbir insan kuyruğuna düşmüyordu — D1 tam olarak bunu açmak zorunda.
# --------------------------------------------------------------------------


def test_apply_decision_marks_verification_kind_human_on_verify():
    row = q_dict(verification_kind="model-cross-check")
    out = apply_decision(row, "e", "", by="baran")
    assert out["verification_kind"] == "human"


def test_apply_decision_marks_verification_kind_human_on_reject():
    row = q_dict(verification_kind="model-cross-check")
    out = apply_decision(row, "h", "yanlış sayfa", by="baran")
    assert out["verification_kind"] == "human"


def test_apply_decision_skip_leaves_verification_kind_untouched():
    row = q_dict(verification_kind="model-cross-check")
    out = apply_decision(row, "a", "", by="baran")
    assert out["verification_kind"] == "model-cross-check"


def test_select_review_queue_returns_rows_not_yet_human_verified():
    rows = [
        q_dict(question_id="c001", verification_kind="model-cross-check"),
        q_dict(question_id="c002", verification_kind="human"),
    ]
    queue = select_review_queue(rows)
    assert [r["question_id"] for r in queue] == ["c001"]


def test_select_review_queue_filters_by_slice():
    rows = [
        q_dict(question_id="c001", slice="paraphrase", verification_kind="model-cross-check"),
        q_dict(question_id="c002", slice="tablo-layout", verification_kind="model-cross-check"),
    ]
    queue = select_review_queue(rows, slices={"paraphrase"})
    assert [r["question_id"] for r in queue] == ["c001"]


def test_select_review_queue_includes_mechanical_rows():
    """Mekanik doğrulama en zayıf türdür; insan kuyruğundan muaf değildir."""
    rows = [q_dict(question_id="c001", verification_kind="mechanical:manifest-absence")]
    assert len(select_review_queue(rows)) == 1


def test_select_review_queue_preserves_input_order():
    rows = [
        q_dict(question_id="c003", verification_kind="model-cross-check"),
        q_dict(question_id="c001", verification_kind="model-cross-check"),
    ]
    assert [r["question_id"] for r in select_review_queue(rows)] == ["c003", "c001"]


def test_display_path_is_relative_inside_repo():
    from review_server import REPO_ROOT, display_path

    assert display_path(REPO_ROOT / "data" / "bench" / "x.jsonl", REPO_ROOT) == "data/bench/x.jsonl"


def test_display_path_falls_back_to_absolute_outside_repo():
    """--bench repo dışını gösterebilir (duman testi /tmp kullandı); patlamamalı."""
    from review_server import REPO_ROOT, display_path

    assert display_path(Path("/tmp/smoke.jsonl"), REPO_ROOT) == "/tmp/smoke.jsonl"
