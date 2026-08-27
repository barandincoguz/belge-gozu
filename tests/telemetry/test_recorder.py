import json
import sqlite3
import threading
from pathlib import Path

from belge_gozu.telemetry.recorder import EventRecorder
from belge_gozu.telemetry.schema import RequestEvent


def _ev(i: int = 0) -> RequestEvent:
    return RequestEvent(
        ts=f"2026-08-26T00:00:{i:02d}+00:00",
        endpoint="/search",
        status="ok",
        http_status=200,
        total_ms=float(i),
        query_sha256="c" * 64,
        detail={"i": i},
    )


def test_record_roundtrip(tmp_path: Path):
    rec = EventRecorder(tmp_path / "t.sqlite")
    rec.record(_ev(1))
    row = (
        sqlite3.connect(tmp_path / "t.sqlite")
        .execute("SELECT endpoint, status, total_ms, detail FROM events")
        .fetchone()
    )
    assert row[0] == "/search" and row[1] == "ok" and row[2] == 1.0
    assert json.loads(row[3]) == {"i": 1}
    rec.close()


def test_wal_mode_enabled(tmp_path: Path):
    rec = EventRecorder(tmp_path / "t.sqlite")
    mode = sqlite3.connect(tmp_path / "t.sqlite").execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    rec.close()


def test_concurrent_writes_all_land(tmp_path: Path):
    rec = EventRecorder(tmp_path / "t.sqlite")
    threads = [
        threading.Thread(target=lambda i=i: [rec.record(_ev(i)) for _ in range(20)])
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    n = sqlite3.connect(tmp_path / "t.sqlite").execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert n == 160
    rec.close()


def test_record_never_raises(tmp_path: Path, caplog):
    rec = EventRecorder(tmp_path / "t.sqlite")
    rec._db.close()  # bağlantıyı boz — record yine de sessiz kalmalı
    rec.record(_ev())  # exception yok


_OLD_EVENTS_DDL = """CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  status TEXT NOT NULL,
  http_status INTEGER NOT NULL,
  total_ms REAL NOT NULL,
  encode_ms REAL, stage1_ms REAL, stage2_ms REAL, answer_ms REAL,
  top_score REAL, margin_1_2 REAL,
  abstained INTEGER, honest_miss INTEGER,
  k INTEGER, candidates INTEGER,
  query_len INTEGER NOT NULL DEFAULT 0,
  query_text TEXT,
  query_sha256 TEXT NOT NULL,
  answer_len INTEGER, citations_n INTEGER,
  tokens_in INTEGER, tokens_out INTEGER, tokens_per_s REAL, est_cost_usd REAL,
  error_type TEXT,
  detail TEXT NOT NULL DEFAULT '{}'
)"""


def test_migration_adds_new_columns_to_old_table(tmp_path: Path):
    db_path = tmp_path / "old.sqlite"
    old_db = sqlite3.connect(db_path)
    old_db.execute(_OLD_EVENTS_DDL)
    old_db.commit()
    old_db.close()

    rec = EventRecorder(db_path)  # migrasyon burada çalışmalı, hata fırlatmamalı
    cols = {r[1] for r in rec._db.execute("PRAGMA table_info(events)")}
    assert "pipeline" in cols and "index_revision" in cols

    ev = _ev(1)
    ev.pipeline = "exhaustive"
    ev.index_revision = "abc123456789/cpe-0.3.18/sign-1bit"
    rec.record(ev)  # exception yok
    row = sqlite3.connect(db_path).execute("SELECT pipeline, index_revision FROM events").fetchone()
    assert row == ("exhaustive", "abc123456789/cpe-0.3.18/sign-1bit")
    rec.close()
