import sqlite3

from belge_gozu.telemetry.schema import EVENTS_DDL, EVENTS_INDEXES, RequestEvent


def test_request_event_minimal_and_full():
    ev = RequestEvent(
        ts="2026-08-26T00:00:00+00:00",
        endpoint="/search",
        status="ok",
        http_status=200,
        total_ms=12.5,
        query_sha256="a" * 64,
    )
    assert ev.encode_ms is None and ev.detail == {}
    full = RequestEvent(
        ts="t",
        endpoint="/ask",
        status="answered",
        http_status=200,
        total_ms=9000.0,
        encode_ms=1500.0,
        stage1_ms=8.0,
        stage2_ms=40.0,
        answer_ms=7000.0,
        top_score=60.9,
        margin_1_2=0.2,
        abstained=False,
        honest_miss=False,
        k=5,
        candidates=200,
        query_len=42,
        query_text="soru",
        query_sha256="b" * 64,
        answer_len=300,
        citations_n=1,
        tokens_in=5000,
        tokens_out=210,
        tokens_per_s=30.0,
        est_cost_usd=0.00058,
        detail={"hits": [{"page_id": "k1:1", "score": 60.9}]},
    )
    assert full.tokens_out == 210


def test_ddl_creates_table_and_indexes():
    db = sqlite3.connect(":memory:")
    db.execute(EVENTS_DDL)
    for idx in EVENTS_INDEXES:
        db.execute(idx)
    cols = {r[1] for r in db.execute("PRAGMA table_info(events)")}
    assert {
        "ts",
        "endpoint",
        "status",
        "total_ms",
        "top_score",
        "tokens_out",
        "query_sha256",
        "detail",
        "pipeline",
        "index_revision",
    } <= cols


def test_request_event_accepts_pipeline_and_index_revision():
    ev = RequestEvent(
        ts="t",
        endpoint="/search",
        status="ok",
        http_status=200,
        total_ms=1.0,
        query_sha256="e" * 64,
        pipeline="exhaustive",
        index_revision="abc123456789/cpe-0.3.18/sign-1bit",
    )
    assert ev.pipeline == "exhaustive"
    assert ev.index_revision == "abc123456789/cpe-0.3.18/sign-1bit"


def test_ddl_and_model_carry_score_scale(tmp_path=None):
    """Y18: `top_score`ın HANGİ ÖLÇEKTE olduğu satırın kendisinde yazar."""
    db = sqlite3.connect(":memory:")
    db.execute(EVENTS_DDL)
    cols = {r[1] for r in db.execute("PRAGMA table_info(events)")}
    assert "score_scale" in cols
    ev = RequestEvent(
        ts="t",
        endpoint="/ask",
        status="answered",
        http_status=200,
        total_ms=1.0,
        query_sha256="f" * 64,
        pipeline="hybrid",
        score_scale="hybrid-bm25",
    )
    assert ev.score_scale == "hybrid-bm25"
    assert (
        RequestEvent(
            ts="t",
            endpoint="/ask",
            status="ok",
            http_status=200,
            total_ms=1.0,
            query_sha256="g" * 64,
        ).score_scale
        is None
    )  # etiketsiz geçmiş NULL kalır


def test_column_lists_stay_in_sync():
    """L3: kolon adı DÖRT yerde paralel tutuluyor — bir tanesi kayarsa sessiz veri kaybı.

    `EVENTS_DDL` (tablo), `RequestEvent` (model), `_COLUMNS` (INSERT listesi) ve
    `_ADDED_COLUMNS` (eski tablolara migrasyon). `_COLUMNS`'ta olmayan bir kolon
    INSERT'e hiç girmez ve **sessizce NULL kalır** — hata da vermez. Bu test o
    sınıfı kapatıyor; migrasyon listesi ayrıca DDL'in bir ALT KÜMESİ olmalı.
    """
    from belge_gozu.telemetry.recorder import _ADDED_COLUMNS, _COLUMNS

    db = sqlite3.connect(":memory:")
    db.execute(EVENTS_DDL)
    ddl_cols = [r[1] for r in db.execute("PRAGMA table_info(events)")]

    # `id` otomatik artan birincil anahtar: tabloda var, yazılmaz, modelde yok.
    assert ddl_cols[0] == "id"
    assert _COLUMNS == ddl_cols[1:], "INSERT listesi DDL sırasıyla birebir olmalı"
    assert set(_COLUMNS) == set(RequestEvent.model_fields), "model <-> INSERT ayrışmış"
    assert {c for c, _ in _ADDED_COLUMNS} <= set(ddl_cols), "migrasyon DDL'de olmayan kolon ekliyor"
