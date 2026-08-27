import json
from pathlib import Path

import pytest

from belge_gozu.bench.dataset import (
    BenchQuestion,
    bench_stats,
    load_bench,
    load_splits,
    question_split,
)


def q_dict(**over) -> dict:
    base = dict(
        question_id="q1",
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
        verified_by="baran",
        verification_status="verified",
    )
    base.update(over)
    return base


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_load_verified_only(tmp_path: Path):
    p = tmp_path / "b.jsonl"
    write_jsonl(p, [q_dict(), q_dict(question_id="q2", verification_status="draft")])
    assert [q.question_id for q in load_bench(p)] == ["q1"]
    assert len(load_bench(p, only_verified=False)) == 2


def test_answerable_requires_gold_pages():
    with pytest.raises(ValueError):
        BenchQuestion(**q_dict(gold_page_ids=[]))


def test_answerable_requires_reference_answer():
    with pytest.raises(ValueError):
        BenchQuestion(**q_dict(reference_answer=""))


def test_answerable_forbids_unanswerable_reason():
    with pytest.raises(ValueError):
        BenchQuestion(**q_dict(unanswerable_reason="korpus-disi"))


def test_unanswerable_requires_reason_and_no_gold():
    with pytest.raises(ValueError):
        BenchQuestion(
            **q_dict(
                answerable=False, gold_page_ids=[], reference_answer="", unanswerable_reason=None
            )
        )
    ok = BenchQuestion(
        **q_dict(
            answerable=False,
            gold_doc_ids=[],
            gold_page_ids=[],
            gold_article_ids=[],
            minimal_evidence_spans=[],
            reference_answer="",
            slice="korpus-disi",
            unanswerable_reason="korpus-disi",
        )
    )
    assert ok.answerable is False


def test_unanswerable_forbids_nonempty_gold_pages():
    with pytest.raises(ValueError):
        BenchQuestion(
            **q_dict(
                answerable=False,
                gold_page_ids=["k4721:4"],
                reference_answer="",
                unanswerable_reason="korpus-disi",
            )
        )


def test_gold_page_doc_consistency():
    with pytest.raises(ValueError):
        BenchQuestion(**q_dict(gold_page_ids=["k9999:1"]))


def test_gold_page_id_requires_colon():
    with pytest.raises(ValueError):
        BenchQuestion(**q_dict(gold_page_ids=["k4721"]))


def test_verified_requires_verified_by():
    with pytest.raises(ValueError):
        BenchQuestion(**q_dict(verified_by=""))


def test_verification_note_defaults_to_empty_string():
    q = BenchQuestion(**q_dict())
    assert q.verification_note == ""


def test_verification_note_roundtrips_when_set():
    q = BenchQuestion(**q_dict(verification_note="h yanlış sayfa"))
    assert q.verification_note == "h yanlış sayfa"


def test_bench_stats_counts_all_slices_with_zero_default():
    qs = [BenchQuestion(**q_dict()), BenchQuestion(**q_dict(question_id="q2"))]
    stats = bench_stats(qs)
    assert stats["paraphrase"] == 2
    assert stats["multi-hop"] == 0
    assert len(stats) == 12


def test_split_assignment(tmp_path: Path):
    sp = tmp_path / "splits.json"
    sp.write_text(json.dumps({"dev_docs": ["k4721"], "test_docs": ["k6098"]}))
    splits = load_splits(sp)
    assert question_split(BenchQuestion(**q_dict()), splits) == "dev"
    ood = BenchQuestion(
        **q_dict(
            question_id="ood1",
            answerable=False,
            gold_doc_ids=[],
            gold_page_ids=[],
            gold_article_ids=[],
            minimal_evidence_spans=[],
            reference_answer="",
            slice="anlamsiz-ood",
            unanswerable_reason="anlamsiz",
        )
    )
    # gold_doc_ids boş -> sha256(question_id) tabanlı deterministik atama.
    # sha256("ood1") = f27f8a54...e398, int(...,16) % 2 == 0 -> "dev".
    # (Eski hali `in ("dev","test")` idi: dönüş tipi zaten bu ikisi olduğu için
    # kuralın kendisini değil, hiçbir şeyi doğrulamıyordu.)
    assert question_split(ood, splits) == "dev"


def test_split_assignment_unknown_doc_defaults_to_dev(tmp_path: Path):
    sp = tmp_path / "splits.json"
    sp.write_text(json.dumps({"dev_docs": ["k4721"], "test_docs": ["k6098"]}))
    splits = load_splits(sp)
    q = BenchQuestion(**q_dict(gold_doc_ids=["k9999"], gold_page_ids=["k9999:4"]))
    assert question_split(q, splits) == "dev"
