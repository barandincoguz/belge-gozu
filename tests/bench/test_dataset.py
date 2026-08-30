import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from belge_gozu.bench.dataset import (
    BenchQuestion,
    assign_split,
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


def test_verification_kind_defaults_to_human():
    # Alan eklenmeden önce yazılmış satırlar (verification_kind anahtarı yok)
    # insan doğrulaması sayılmaya devam etmeli.
    q = BenchQuestion(**q_dict())
    assert q.verification_kind == "human"


@pytest.mark.parametrize("kind", ["human", "model-cross-check"])
def test_verification_kind_accepts_both_values(kind: str):
    q = BenchQuestion(**q_dict(verification_kind=kind))
    assert q.verification_kind == kind


def test_verification_kind_rejects_unknown_value():
    with pytest.raises(ValueError):
        BenchQuestion(**q_dict(verification_kind="model"))


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


# --------------------------------------------------------- P2: sözlük genişlemeleri
def unans_dict(**over) -> dict:
    """Cevaplanamaz satır iskeleti (unans_v1.jsonl biçimi)."""
    fields = dict(
        answerable=False,
        gold_doc_ids=[],
        gold_page_ids=[],
        gold_article_ids=[],
        minimal_evidence_spans=[],
        reference_answer="",
        source_type="ajan-taslak",
        verified_by="",
        verification_status="draft",
        verification_note="taslak",
    )
    fields.update(over)
    return q_dict(**fields)


@pytest.mark.parametrize("reason", ["korpus-disi", "eksik-kanit", "anlamsiz", "belirsiz"])
def test_unanswerable_reason_vocabulary(reason: str):
    """Sözlüğü sabitler: `eksik-kanit` (sınırdaki sınıf) kabul edilmeli."""
    q = BenchQuestion(**unans_dict(slice="eksik-kanit", unanswerable_reason=reason))
    assert q.unanswerable_reason == reason


def test_unanswerable_reason_rejects_unknown():
    with pytest.raises(ValidationError):
        BenchQuestion(**unans_dict(slice="eksik-kanit", unanswerable_reason="uydurma"))


def test_source_type_ajan_taslak_is_distinct_from_insan_onayli():
    """`ajan-taslak` (onaysız) ile `ajan-taslak-insan-onayli` ayrı değerlerdir."""
    q = BenchQuestion(**unans_dict(slice="korpus-disi", unanswerable_reason="korpus-disi"))
    assert q.source_type == "ajan-taslak"
    onayli = BenchQuestion(**q_dict(source_type="ajan-taslak-insan-onayli"))
    assert onayli.source_type != q.source_type


def test_verification_kind_mechanical_is_a_third_kind():
    """Mekanik manifest-yokluk doğrulaması insan/model turlarıyla karışmamalı."""
    q = BenchQuestion(
        **unans_dict(
            slice="korpus-disi",
            unanswerable_reason="korpus-disi",
            verified_by="script:validate_unans",
            verification_status="verified",
            verification_kind="mechanical:manifest-absence",
        )
    )
    assert q.verification_kind == "mechanical:manifest-absence"
    assert q.verification_kind not in ("human", "model-cross-check")


def test_verification_kind_rejects_unknown():
    with pytest.raises(ValidationError):
        BenchQuestion(**unans_dict(verification_kind="mechanical:vibes"))


# ------------------------------------------------------------------- assign_split
SPLITS = {"dev_docs": {"k4721", "k4857"}, "test_docs": {"k6098", "k5237"}}


def test_assign_split_answerable_follows_primary_gold_doc():
    assert assign_split(q_dict(), SPLITS) == "dev"
    assert assign_split(q_dict(gold_doc_ids=["k6098"], gold_page_ids=["k6098:1"]), SPLITS) == "test"


def test_assign_split_answerable_unknown_doc_defaults_to_dev():
    q = q_dict(gold_doc_ids=["k9999"], gold_page_ids=["k9999:4"])
    assert assign_split(q, SPLITS) == "dev"


def test_assign_split_korpus_disi_groups_by_anchor_law():
    """Aynı absent kanuna dayanan iki farklı soru AYNI yakaya düşmeli."""
    a = unans_dict(
        question_id="u001",
        slice="korpus-disi",
        unanswerable_reason="korpus-disi",
        _anchor_law="5901",
    )
    b = unans_dict(
        question_id="u002",
        slice="korpus-disi",
        unanswerable_reason="korpus-disi",
        _anchor_law="5901",
    )
    assert assign_split(a, SPLITS) == assign_split(b, SPLITS)
    # farklı kanun farklı yakaya düşebilmeli (kural sabit-değer döndürmüyor)
    others = {
        assign_split(
            unans_dict(
                question_id="u003",
                slice="korpus-disi",
                unanswerable_reason="korpus-disi",
                _anchor_law=no,
            ),
            SPLITS,
        )
        for no in ("5901", "7179", "5682", "6458", "7201", "3402")
    }
    assert others == {"dev", "test"}


def test_assign_split_eksik_kanit_follows_subject_doc():
    """Konu belgesi hukuk-gruplu bölmeyi belirler — cevaplanabilirlerle aynı yaka."""
    dev_row = unans_dict(
        question_id="u261",
        slice="eksik-kanit",
        unanswerable_reason="eksik-kanit",
        _subject_doc="k4857",
    )
    test_row = unans_dict(
        question_id="u262",
        slice="eksik-kanit",
        unanswerable_reason="eksik-kanit",
        _subject_doc="k6098",
    )
    assert assign_split(dev_row, SPLITS) == "dev"
    assert assign_split(test_row, SPLITS) == "test"


def test_assign_split_anlamsiz_uses_question_id_hash():
    row = unans_dict(question_id="u201", slice="anlamsiz-ood", unanswerable_reason="anlamsiz")
    first = assign_split(row, SPLITS)
    assert first in ("dev", "test")
    assert assign_split(row, SPLITS) == first  # deterministik
    # id değişince kural gerçekten id'ye bakıyor olmalı
    variants = {
        assign_split(
            unans_dict(
                question_id=f"u{i:03d}", slice="anlamsiz-ood", unanswerable_reason="anlamsiz"
            ),
            SPLITS,
        )
        for i in range(201, 211)
    }
    assert variants == {"dev", "test"}


def test_assign_split_falls_back_when_underscore_fields_missing():
    """BenchQuestion alt çizgili alanları taşımaz -> qid hash'ine düşülmeli."""
    raw = unans_dict(
        question_id="u001",
        slice="korpus-disi",
        unanswerable_reason="korpus-disi",
        _anchor_law="5901",
    )
    model = BenchQuestion(**{k: v for k, v in raw.items() if not k.startswith("_")})
    assert assign_split(model, SPLITS) in ("dev", "test")
    # çapa bilgisi kaybolduğu için kanun-gruplu sonuçla aynı olmak ZORUNDA değil;
    # önemli olan çökmemesi ve deterministik olması
    assert assign_split(model, SPLITS) == assign_split(model, SPLITS)


def test_assign_split_is_pure_no_file_access(tmp_path: Path):
    """Saf fonksiyon: aynı girdi + aynı splits -> aynı çıktı, yan etki yok."""
    before = sorted(p.name for p in tmp_path.iterdir())
    row = unans_dict(question_id="u205", slice="anlamsiz-ood", unanswerable_reason="anlamsiz")
    results = {assign_split(row, SPLITS) for _ in range(5)}
    assert len(results) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_load_bench_and_splits_accept_str_paths(tmp_path: Path):
    """İnceleme L2: load_bench/load_splits düz `str` yol kabul etmeli (Path'e çevrilir)."""
    bench = tmp_path / "mini.jsonl"
    row = unans_dict(question_id="u001", slice="korpus-disi", unanswerable_reason="korpus-disi")
    bench.write_text(json.dumps(row) + "\n")
    qs = load_bench(str(bench), only_verified=False)
    assert len(qs) == 1
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"seed": 1, "rule": "test", "dev_docs": ["k1"], "test_docs": ["k2"]})
    )
    loaded = load_splits(str(splits))
    assert loaded["test_docs"] == {"k2"}
