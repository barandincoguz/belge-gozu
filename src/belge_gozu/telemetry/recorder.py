import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from belge_gozu.telemetry.schema import EVENTS_DDL, EVENTS_INDEXES, RequestEvent

logger = logging.getLogger(__name__)

# Yazma hatası log kısıtı (Y22): en fazla bu aralıkta BİR satır, ama ASLA
# tamamen susmaz. Eskiden ilk hata WARNING'di ve sonrası SONSUZA DEK sessizdi —
# disk dolduğunda ya da sqlite salt-okunur olduğunda sistem hizmet vermeye
# devam ediyor, `/metrics` normal görünüyor, ama olay tablosu SAATLERCE
# büyümüyordu. P2'nin eğitim kümesinde açılan delik ayrıca RASTGELE DEĞİL:
# tam olarak yükün en yoğun olduğu anda açılır.
WRITE_ERROR_LOG_INTERVAL_S = 60.0

# Eski `events` tablolarına eklenen kolonlar (kolon zaten varsa sqlite hata
# fırlatır, yutuyoruz). Geçmiş satırlar DOLDURULMAZ — bir migrasyon o satırların
# hangi pipeline'da/ölçekte üretildiğini bilemez, uydurmak veriyi bozar.
_ADDED_COLUMNS = (
    ("pipeline", "TEXT"),
    ("index_revision", "TEXT"),
    ("score_scale", "TEXT"),
    ("honest_miss", "INTEGER"),
)

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
    "score_scale",
    "index_revision",
    "detail",
]
_INSERT = (
    f"INSERT INTO events ({', '.join(_COLUMNS)}) VALUES ({', '.join(':' + c for c in _COLUMNS)})"
)


class EventRecorder:
    """WAL'lı, thread-güvenli, best-effort olay yazıcısı.

    Telemetri ilkesi: kayıt hatası hiçbir koşulda isteği düşürmez — ama
    SESSİZ DE KALMAZ: hatalar hız-sınırlı (dakikada bir) loglanır ve iki log
    arasında yutulan hata sayısı satırda raporlanır (Y22).
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._last_warn_at: float | None = None
        self._suppressed = 0
        self.write_failures = 0
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute(EVENTS_DDL)
        # Migrasyon: eski events tablolarına eksik kolonları ekle (best-effort —
        # kolon zaten varsa sqlite hata fırlatır, bunu yutuyoruz; telemetri
        # hiçbir koşulda isteği düşürmez).
        for col, coltype in _ADDED_COLUMNS:
            try:
                self._db.execute(f"ALTER TABLE events ADD COLUMN {col} {coltype}")
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
            self._note_write_failure()

    def _note_write_failure(self) -> None:
        """Hız-sınırlı uyarı: dakikada en fazla bir satır, ama hiç susmaz."""
        now = time.monotonic()
        with self._lock:
            self.write_failures += 1
            due = self._last_warn_at is None or (now - self._last_warn_at) >= (
                WRITE_ERROR_LOG_INTERVAL_S
            )
            if not due:
                self._suppressed += 1
                return
            suppressed, self._suppressed = self._suppressed, 0
            self._last_warn_at = now
            total = self.write_failures
        logger.warning(
            "telemetri olay yazımı başarısız (toplam %d; son uyarıdan beri %d susturuldu; "
            "en fazla %.0f sn'de bir uyarılır)",
            total,
            suppressed,
            WRITE_ERROR_LOG_INTERVAL_S,
            exc_info=True,
        )

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass
