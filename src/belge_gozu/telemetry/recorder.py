import json
import logging
import sqlite3
import threading
from pathlib import Path

from belge_gozu.telemetry.schema import EVENTS_DDL, EVENTS_INDEXES, RequestEvent

logger = logging.getLogger(__name__)

_COLUMNS = [
    "ts",
    "endpoint",
    "status",
    "http_status",
    "total_ms",
    "encode_ms",
    "stage1_ms",
    "stage2_ms",
    "answer_ms",
    "top_score",
    "margin_1_2",
    "abstained",
    "honest_miss",
    "k",
    "candidates",
    "query_len",
    "query_text",
    "query_sha256",
    "answer_len",
    "citations_n",
    "tokens_in",
    "tokens_out",
    "tokens_per_s",
    "est_cost_usd",
    "error_type",
    "pipeline",
    "index_revision",
    "detail",
]
_INSERT = (
    f"INSERT INTO events ({', '.join(_COLUMNS)}) VALUES ({', '.join(':' + c for c in _COLUMNS)})"
)


class EventRecorder:
    """WAL'lı, thread-güvenli, best-effort olay yazıcısı.

    Telemetri ilkesi: kayıt hatası hiçbir koşulda isteği düşürmez —
    ilk hata WARNING olarak loglanır, sonrakiler sessizdir.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._warned = False
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute(EVENTS_DDL)
        # Migrasyon: eski events tablolarına eksik kolonları ekle (best-effort —
        # kolon zaten varsa sqlite hata fırlatır, bunu yutuyoruz; telemetri
        # hiçbir koşulda isteği düşürmez).
        for col in ("pipeline", "index_revision"):
            try:
                self._db.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass
        for idx in EVENTS_INDEXES:
            self._db.execute(idx)
        self._db.commit()

    def record(self, ev: RequestEvent) -> None:
        try:
            row = ev.model_dump()
            row["detail"] = json.dumps(row["detail"], ensure_ascii=False)
            for flag in ("abstained", "honest_miss"):
                if row[flag] is not None:
                    row[flag] = int(row[flag])
            with self._lock:
                self._db.execute(_INSERT, row)
                self._db.commit()
        except Exception:
            if not self._warned:
                logger.warning("telemetri olay yazımı başarısız (bir kez uyarılır)", exc_info=True)
                self._warned = True

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass
