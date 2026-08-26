import json
import sqlite3
import threading
from pathlib import Path

from belge_gozu.telemetry.recorder import EventRecorder
from belge_gozu.telemetry.schema import RequestEvent


def _ev(i: int = 0) -> RequestEvent:
    return RequestEvent(ts=f"2026-08-26T00:00:{i:02d}+00:00", endpoint="/search",
                        status="ok", http_status=200, total_ms=float(i),
                        query_sha256="c" * 64, detail={"i": i})


def test_record_roundtrip(tmp_path: Path):
    rec = EventRecorder(tmp_path / "t.sqlite")
    rec.record(_ev(1))
    row = sqlite3.connect(tmp_path / "t.sqlite").execute(
        "SELECT endpoint, status, total_ms, detail FROM events").fetchone()
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
    threads = [threading.Thread(target=lambda i=i: [rec.record(_ev(i)) for _ in range(20)])
               for i in range(8)]
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
