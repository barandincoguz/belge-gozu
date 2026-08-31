import json
import logging
import sqlite3
import threading
from pathlib import Path

from belge_gozu.telemetry.recorder import WRITE_ERROR_LOG_INTERVAL_S, EventRecorder
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


def test_unwritable_parent_falls_back_to_memory(tmp_path: Path, caplog, monkeypatch):
    requested = tmp_path / "readonly" / "requests.sqlite"
    real_mkdir = Path.mkdir

    def deny_requested_parent(path: Path, *args, **kwargs):
        if path == requested.parent:
            raise PermissionError("read-only filesystem")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_requested_parent)

    with caplog.at_level(logging.WARNING):
        rec = EventRecorder(requested)
    rec.record(_ev(1))

    assert rec.db_path == requested
    assert rec._db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert caplog.text.count("kalıcı telemetri açılamadı; bellek içi kayda düşülüyor") == 1
    rec.close()


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


# --- Y18: score_scale migrasyonu --------------------------------------------


def test_migration_adds_score_scale_to_old_table(tmp_path: Path):
    """Yeni kolon eski tabloya eklenir; GEÇMİŞ satırlar NULL KALIR.

    Bir migrasyon o satırların hangi ölçekte üretildiğini BİLEMEZ — doldurmak
    veriyi bozardı (üretimde `top_score` sütununda üç uyumsuz ölçek karışık
    duruyor ve satırların %96'sı etiketsiz)."""
    db_path = tmp_path / "old.sqlite"
    old = sqlite3.connect(db_path)
    old.execute(_OLD_EVENTS_DDL)
    old.execute(
        "INSERT INTO events (ts, endpoint, status, http_status, total_ms, query_sha256, top_score)"
        " VALUES ('eski', '/search', 'ok', 200, 1.0, 'x', 68.39)"
    )
    old.commit()
    old.close()

    rec = EventRecorder(db_path)
    cols = {r[1] for r in rec._db.execute("PRAGMA table_info(events)")}
    assert {"score_scale", "honest_miss", "pipeline", "index_revision"} <= cols

    ev = _ev(1)
    ev.score_scale = "hybrid-bm25"
    rec.record(ev)
    rows = sqlite3.connect(db_path).execute("SELECT ts, score_scale FROM events").fetchall()
    assert rows == [("eski", None), (ev.ts, "hybrid-bm25")]
    rec.close()


# --- Y22: yazma hataları hız-sınırlı loglanır, ASLA sonsuza dek susmaz -------


class _FailingDB:
    """Diski dolmuş / salt-okunur sqlite: her yazma denemesi patlar."""

    def execute(self, *a, **kw):
        raise sqlite3.OperationalError("attempt to write a readonly database")

    def commit(self) -> None: ...

    def close(self) -> None: ...


def _failing(tmp_path: Path) -> EventRecorder:
    rec = EventRecorder(tmp_path / "t.sqlite")
    rec._db.close()
    rec._db = _FailingDB()  # type: ignore[assignment]
    return rec


def test_write_failures_are_logged_at_most_once_per_interval(tmp_path: Path, caplog, monkeypatch):
    """Eskiden ilk hata WARNING'di ve sonrası SONSUZA DEK sessizdi: disk
    dolduğunda sistem hizmet vermeye devam ediyor, `/metrics` normal
    görünüyor, ama olay tablosu saatlerce büyümüyordu."""
    clock = {"t": 1000.0}
    monkeypatch.setattr("belge_gozu.telemetry.recorder.time.monotonic", lambda: clock["t"])
    rec = _failing(tmp_path)
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            rec.record(_ev())  # exception YOK
        assert caplog.text.count("telemetri olay yazımı başarısız") == 1
        clock["t"] += WRITE_ERROR_LOG_INTERVAL_S + 1  # pencere dolsun
        rec.record(_ev())
    assert caplog.text.count("telemetri olay yazımı başarısız") == 2
    assert rec.write_failures == 6
    rec.close()


def test_second_warning_reports_the_suppressed_count(tmp_path: Path, caplog, monkeypatch):
    """Susturulan hata sayısı KAYBOLMAZ — delik büyüklüğü loglanır."""
    clock = {"t": 0.0}
    monkeypatch.setattr("belge_gozu.telemetry.recorder.time.monotonic", lambda: clock["t"])
    rec = _failing(tmp_path)
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            rec.record(_ev())
        clock["t"] += WRITE_ERROR_LOG_INTERVAL_S
        rec.record(_ev())
    assert "son uyarıdan beri 3 susturuldu" in caplog.text
    assert "toplam 5" in caplog.text
    rec.close()
