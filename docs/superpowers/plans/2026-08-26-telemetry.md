# Belge-Gözü Telemetri Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Her isteği aşama kırılımı + token/maliyet bilgisiyle SQLite `events` tablosuna ve Prometheus'a kaydeden merkezî telemetri; OrbStack'te Grafana dashboard'u; loadgen + `docs/research/` bulgu altyapısı.

**Architecture:** `telemetry/` paketi (schema/collect/recorder/prom/export) + contextvar tabanlı StageCollector ile mevcut modüllere minimal dokunuş. Uygulama tek başına tam çalışır; Prometheus+Grafana yalnız lokal compose. Ham olay = gerçeğin kaynağı; Prometheus = görünüm.

**Tech Stack:** prometheus-client (yeni ana bağımlılık), sqlite WAL, pyarrow (mevcut), httpx+asyncio (loadgen), Prometheus v2.53 + Grafana 11.1 (compose, lokal).

**Spec:** `docs/superpowers/specs/2026-08-26-telemetry-design.md`

## Global Constraints

- CI'da test ağ/GPU/model kullanmaz: `uv run pytest -m "not slow" -q` her task sonunda geçer.
- `uv run ruff check .` ve `uv run pyright` her task sonunda temiz.
- Telemetri **asla** isteği düşürmez: recorder/olay birleştirme hataları yutulur (bir kez WARNING).
- Hiçbir test/CI adımı Gemini kotası (≈20 çağrı/gün) yakmaz; loadgen varsayılanı `/search`.
- Her task kendi commit'iyle biter; commit mesajları Türkçe, gövdede kısa açıklama.
- `.env`/anahtarlar asla yazdırılmaz/commit'lenmez.
- Commit imzası: her commit mesajının sonuna şu iki satır eklenir:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` ve
  `Claude-Session: https://claude.ai/code/session_01G4ezdAg4Mq8SwQXsrFWegb`

## File Structure

```
src/belge_gozu/telemetry/__init__.py   # boş; paket işareti
src/belge_gozu/telemetry/schema.py    # RequestEvent + EVENTS_DDL + EVENTS_INDEXES
src/belge_gozu/telemetry/collect.py   # StageCollector, collecting(), stage(), annotate()
src/belge_gozu/telemetry/recorder.py  # EventRecorder (WAL, Lock, best-effort)
src/belge_gozu/telemetry/prom.py      # PromMetrics (registry, observe, render, inflight)
src/belge_gozu/telemetry/export.py    # export_events(db, out, fmt)
scripts/loadgen.py                    # async yük üreticisi
scripts/queries_sample.txt            # 30 örnek Türkçe mevzuat sorusu
observability/docker-compose.yml
observability/prometheus.yml
observability/grafana/provisioning/datasources/prometheus.yml
observability/grafana/provisioning/dashboards/provider.yml
observability/grafana/provisioning/dashboards/belge-gozu.json
docs/research/metrics-catalog.md
docs/research/runbook.md
docs/research/findings/2026-08-26-baseline.md
docs/research/figures/               # (görseller Task 10'da eklenir)
tests/telemetry/{__init__,test_schema,test_collect,test_recorder,test_prom,test_export}.py
tests/test_loadgen.py
```

Dokunulan mevcut dosyalar: `config.py`, `answer/gemini.py`, `answer/base.py`,
`retrieval/core.py`, `app/main.py`, `cli.py`, `pyproject.toml`, `Makefile`,
`tests/app/test_api.py`, `tests/answer/test_gemini.py`, `README.md` (Task 10).

---

### Task 1: Olay şeması + StageCollector (`schema.py`, `collect.py`)

**Files:**
- Create: `src/belge_gozu/telemetry/__init__.py` (boş), `src/belge_gozu/telemetry/schema.py`, `src/belge_gozu/telemetry/collect.py`
- Test: `tests/telemetry/__init__.py` (boş), `tests/telemetry/test_schema.py`, `tests/telemetry/test_collect.py`

**Interfaces:**
- Consumes: yok (stdlib + pydantic)
- Produces:
  - `RequestEvent` (pydantic BaseModel; alanlar aşağıda — sonraki tüm task'lar bunu kullanır)
  - `EVENTS_DDL: str`, `EVENTS_INDEXES: list[str]`
  - `StageCollector` (`.stages: dict[str, float]` ms cinsinden, `.notes: dict[str, object]`)
  - `collecting() -> ContextManager[StageCollector]`, `stage(name: str) -> ContextManager[None]`, `annotate(key: str, value: object) -> None`

- [ ] **Step 1: Başarısız testleri yaz**

`tests/telemetry/test_schema.py`:

```python
import sqlite3

from belge_gozu.telemetry.schema import EVENTS_DDL, EVENTS_INDEXES, RequestEvent


def test_request_event_minimal_and_full():
    ev = RequestEvent(ts="2026-08-26T00:00:00+00:00", endpoint="/search", status="ok",
                      http_status=200, total_ms=12.5, query_sha256="a" * 64)
    assert ev.encode_ms is None and ev.detail == {}
    full = RequestEvent(ts="t", endpoint="/ask", status="answered", http_status=200,
                        total_ms=9000.0, encode_ms=1500.0, stage1_ms=8.0, stage2_ms=40.0,
                        answer_ms=7000.0, top_score=60.9, margin_1_2=0.2, abstained=False,
                        honest_miss=False, k=5, candidates=200, query_len=42,
                        query_text="soru", query_sha256="b" * 64, answer_len=300,
                        citations_n=1, tokens_in=5000, tokens_out=210, tokens_per_s=30.0,
                        est_cost_usd=0.00058, detail={"hits": [{"page_id": "k1:1", "score": 60.9}]})
    assert full.tokens_out == 210


def test_ddl_creates_table_and_indexes():
    db = sqlite3.connect(":memory:")
    db.execute(EVENTS_DDL)
    for idx in EVENTS_INDEXES:
        db.execute(idx)
    cols = {r[1] for r in db.execute("PRAGMA table_info(events)")}
    assert {"ts", "endpoint", "status", "total_ms", "top_score", "tokens_out",
            "query_sha256", "detail"} <= cols
```

`tests/telemetry/test_collect.py`:

```python
import threading

from belge_gozu.telemetry.collect import annotate, collecting, stage


def test_stage_noop_without_collector():
    with stage("query_encode"):
        pass  # kolektör yok; hata da yok, kayıt da yok
    annotate("tokens_in", 5)  # sessiz no-op


def test_collecting_captures_stages_and_notes():
    with collecting() as col:
        with stage("query_encode"):
            pass
        with stage("answerer"):
            annotate("tokens_out", 42)
    assert set(col.stages) == {"query_encode", "answerer"}
    assert col.stages["query_encode"] >= 0.0  # ms
    assert col.notes == {"tokens_out": 42}


def test_collectors_are_isolated_across_threads():
    seen: dict[str, set[str]] = {}

    def worker(name: str):
        with collecting() as col:
            with stage(name):
                pass
            seen[name] = set(col.stages)

    ts = [threading.Thread(target=worker, args=(f"s{i}",)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert seen == {f"s{i}": {f"s{i}"} for i in range(4)}
```

- [ ] **Step 2: Koş, FAIL gör** — `uv run pytest tests/telemetry/ -q` → ModuleNotFoundError beklenir.

- [ ] **Step 3: Uygula**

`src/belge_gozu/telemetry/schema.py`:

```python
from pydantic import BaseModel, Field

EVENTS_DDL = """CREATE TABLE IF NOT EXISTS events (
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

EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_endpoint_ts ON events(endpoint, ts)",
]


class RequestEvent(BaseModel):
    """Tek isteğin ham olay kaydı. Spec §5'in birebir karşılığı."""

    ts: str
    endpoint: str
    status: str
    http_status: int
    total_ms: float
    encode_ms: float | None = None
    stage1_ms: float | None = None
    stage2_ms: float | None = None
    answer_ms: float | None = None
    top_score: float | None = None
    margin_1_2: float | None = None
    abstained: bool | None = None
    honest_miss: bool | None = None
    k: int | None = None
    candidates: int | None = None
    query_len: int = 0
    query_text: str | None = None
    query_sha256: str = ""
    answer_len: int | None = None
    citations_n: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_per_s: float | None = None
    est_cost_usd: float | None = None
    error_type: str | None = None
    detail: dict = Field(default_factory=dict)
```

`src/belge_gozu/telemetry/collect.py`:

```python
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class StageCollector:
    """İstek boyunca aşama sürelerini (ms) ve notları biriktirir."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.notes: dict[str, object] = {}


_collector: ContextVar[StageCollector | None] = ContextVar("bg_collector", default=None)


@contextmanager
def collecting() -> Iterator[StageCollector]:
    col = StageCollector()
    token = _collector.set(col)
    try:
        yield col
    finally:
        _collector.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    col = _collector.get()
    if col is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        col.stages[name] = (time.perf_counter() - t0) * 1000.0


def annotate(key: str, value: object) -> None:
    col = _collector.get()
    if col is not None:
        col.notes[key] = value
```

`src/belge_gozu/telemetry/__init__.py` ve `tests/telemetry/__init__.py`: boş dosyalar.

- [ ] **Step 4: Koş, PASS gör** — `uv run pytest tests/telemetry/ -q`
- [ ] **Step 5: Tam süit + lint** — `uv run pytest -m "not slow" -q && uv run ruff check . && uv run pyright`
- [ ] **Step 6: Commit** — `git add src/belge_gozu/telemetry tests/telemetry && git commit -m "feat(telemetry): olay şeması + contextvar StageCollector"`

---

### Task 2: EventRecorder (`recorder.py`)

**Files:**
- Create: `src/belge_gozu/telemetry/recorder.py`
- Test: `tests/telemetry/test_recorder.py`

**Interfaces:**
- Consumes: `RequestEvent`, `EVENTS_DDL`, `EVENTS_INDEXES` (Task 1)
- Produces: `EventRecorder(db_path: Path)`; `.record(ev: RequestEvent) -> None` (asla fırlatmaz); `.close() -> None`

- [ ] **Step 1: Başarısız testleri yaz** — `tests/telemetry/test_recorder.py`:

```python
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
```

- [ ] **Step 2: FAIL gör** — `uv run pytest tests/telemetry/test_recorder.py -q`
- [ ] **Step 3: Uygula** — `src/belge_gozu/telemetry/recorder.py`:

```python
import json
import logging
import sqlite3
import threading
from pathlib import Path

from belge_gozu.telemetry.schema import EVENTS_DDL, EVENTS_INDEXES, RequestEvent

logger = logging.getLogger(__name__)

_COLUMNS = [
    "ts", "endpoint", "status", "http_status", "total_ms",
    "encode_ms", "stage1_ms", "stage2_ms", "answer_ms",
    "top_score", "margin_1_2", "abstained", "honest_miss",
    "k", "candidates", "query_len", "query_text", "query_sha256",
    "answer_len", "citations_n", "tokens_in", "tokens_out",
    "tokens_per_s", "est_cost_usd", "error_type", "detail",
]
_INSERT = (
    f"INSERT INTO events ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join(':' + c for c in _COLUMNS)})"
)


class EventRecorder:
    """WAL'lı, thread-güvenli, best-effort olay yazıcısı.

    Telemetri ilkesi: kayıt hatası hiçbir koşulda isteği düşürmez —
    ilk hata WARNING olarak loglanır, sonrakiler sessizdir.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._warned = False
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute(EVENTS_DDL)
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
```

- [ ] **Step 4: PASS gör** — `uv run pytest tests/telemetry/test_recorder.py -q`
- [ ] **Step 5: Tam süit + lint** — Global Constraints komutları
- [ ] **Step 6: Commit** — `git commit -m "feat(telemetry): WAL'lı best-effort EventRecorder"`

---

### Task 3: Gemini token yakalama + fiyat config'i

**Files:**
- Modify: `src/belge_gozu/answer/gemini.py`, `src/belge_gozu/config.py`
- Test: `tests/answer/test_gemini.py` (mevcut dosyaya ekleme/uyarlama)

**Interfaces:**
- Consumes: `annotate` (Task 1)
- Produces:
  - `GenResult` dataclass: `text: str`, `tokens_in: int | None = None`, `tokens_out: int | None = None`
  - `GeminiClient.generate(prompt, images) -> GenResult` (ÖNCEDEN `str` dönüyordu)
  - `GeminiAnswerer.answer(...)` değişmez imza; içeride `annotate("tokens_in"/"tokens_out", …)` çağırır
  - `Settings.gemini_price_in_usd_per_1m: float = 0.10`, `Settings.gemini_price_out_usd_per_1m: float = 0.40`
    (tahmin; env `BG_GEMINI_PRICE_IN_USD_PER_1M` vb. ile geçersiz kılınır; Task 10 runbook'unda güncel fiyat doğrulama adımı var)

- [ ] **Step 1: Mevcut testleri oku** — `tests/answer/test_gemini.py`'daki stub'lar `generate` dönüşünü `str` varsayıyorsa `GenResult`'a uyarlanacak; önce oku, sonra yeni testleri ekle:

```python
from belge_gozu.answer.gemini import GeminiAnswerer, GenResult
from belge_gozu.telemetry.collect import collecting


class StubClient:
    def generate(self, prompt, images):
        return GenResult(text="cevap [S1]", tokens_in=1234, tokens_out=56)


class StubClientNoUsage:
    def generate(self, prompt, images):
        return GenResult(text="cevap [S1]")


def _pages():
    from belge_gozu.retrieval.types import PageHit
    return [PageHit(page_id="k1:1", score=61.0, doc_name="X", page_no=1,
                    image_path="images/x/0001.webp", source_url="u")]


def test_answer_annotates_token_usage():
    ans = GeminiAnswerer("m", "k", client=StubClient())
    with collecting() as col:
        ans.answer("soru", _pages(), lambda p: b"img")
    assert col.notes["tokens_in"] == 1234 and col.notes["tokens_out"] == 56


def test_answer_without_usage_annotates_nothing():
    ans = GeminiAnswerer("m", "k", client=StubClientNoUsage())
    with collecting() as col:
        ans.answer("soru", _pages(), lambda p: b"img")
    assert "tokens_in" not in col.notes and "tokens_out" not in col.notes
```

- [ ] **Step 2: FAIL gör** — `uv run pytest tests/answer/ -q`
- [ ] **Step 3: Uygula** — `gemini.py` değişiklikleri:

```python
from dataclasses import dataclass


@dataclass
class GenResult:
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None
```

`GeminiClient.generate` sonu şöyle olur (resp.text yerine):

```python
        resp = client.models.generate_content(model=self.model, contents=[*parts, prompt])
        usage = getattr(resp, "usage_metadata", None)
        return GenResult(
            text=resp.text or "",
            tokens_in=getattr(usage, "prompt_token_count", None),
            tokens_out=getattr(usage, "candidates_token_count", None),
        )
```

`GeminiAnswerer.answer` içinde `text = self._client.generate(...)` satırı:

```python
        gen = self._client.generate(prompt, images)
        text = gen.text
        if gen.tokens_in is not None:
            annotate("tokens_in", gen.tokens_in)
        if gen.tokens_out is not None:
            annotate("tokens_out", gen.tokens_out)
```

(import: `from belge_gozu.telemetry.collect import annotate`)

`config.py`'ye iki alan (mevcut alanların yanına, yorumla):

```python
    # Tahmini birim fiyatlar (USD / 1M token). Kesin değildir; runbook'taki
    # doğrulama adımıyla güncellenir, env ile geçersiz kılınır.
    gemini_price_in_usd_per_1m: float = 0.10
    gemini_price_out_usd_per_1m: float = 0.40
```

- [ ] **Step 4: PASS + eski testleri uyarla** — `uv run pytest tests/answer/ -q`; `generate`'in `str` döndüğünü varsayan mevcut stub varsa `GenResult(text=...)` yap.
- [ ] **Step 5: Tam süit + lint**
- [ ] **Step 6: Commit** — `git commit -m "feat(telemetry): Gemini usage_metadata yakalama + fiyat config'i"`

---

### Task 4: Prometheus katmanı (`prom.py`)

**Files:**
- Create: `src/belge_gozu/telemetry/prom.py`
- Modify: `pyproject.toml` (dependencies'e `"prometheus-client>=0.20",`)
- Test: `tests/telemetry/test_prom.py`

**Interfaces:**
- Consumes: `RequestEvent` (Task 1)
- Produces: `PromMetrics()` — her çağrı KENDİ `CollectorRegistry`'sini kurar (test/app izolasyonu):
  - `.observe(ev: RequestEvent) -> None`
  - `.inflight(endpoint: str) -> ContextManager[None]`
  - `.set_app_info(*, pages: int, retriever_model: str, gemini_model: str, device: str, version: str, threshold: float) -> None`
  - `.render() -> tuple[bytes, str]` — (gövde, content-type)

- [ ] **Step 1: Başarısız testleri yaz** — `tests/telemetry/test_prom.py`:

```python
from belge_gozu.telemetry.prom import PromMetrics
from belge_gozu.telemetry.schema import RequestEvent


def _ask_ev(**kw) -> RequestEvent:
    base = dict(ts="t", endpoint="/ask", status="answered", http_status=200,
                total_ms=9000.0, encode_ms=1500.0, stage1_ms=8.0, stage2_ms=40.0,
                answer_ms=7000.0, top_score=60.9, margin_1_2=0.2, abstained=False,
                tokens_in=5000, tokens_out=210, tokens_per_s=30.0,
                est_cost_usd=0.0006, query_sha256="d" * 64)
    base.update(kw)
    return RequestEvent(**base)


def test_observe_and_render_contains_series():
    pm = PromMetrics()
    pm.set_app_info(pages=4222, retriever_model="colsmol", gemini_model="gf",
                    device="cpu", version="0.1.0", threshold=60.0)
    pm.observe(_ask_ev())
    pm.observe(_ask_ev(status="abstained", abstained=True, answer_ms=None,
                       tokens_in=None, tokens_out=None, tokens_per_s=None,
                       est_cost_usd=None))
    body, ctype = pm.render()
    text = body.decode()
    assert 'bg_http_requests_total{endpoint="/ask",status="answered"} 1.0' in text
    assert 'bg_abstain_total{reason="threshold"} 1.0' in text
    assert "bg_request_duration_seconds_bucket" in text
    assert 'bg_stage_duration_seconds_bucket{le="2.0",stage="query_encode"}' in text
    assert 'bg_llm_tokens_total{direction="output"} 210.0' in text
    assert "bg_index_pages 4222.0" in text
    assert "openmetrics" in ctype or "text/plain" in ctype


def test_degraded_maps_to_degraded_reason():
    pm = PromMetrics()
    pm.observe(_ask_ev(status="degraded", abstained=True))
    assert 'bg_abstain_total{reason="degraded"} 1.0' in pm.render()[0].decode()


def test_inflight_gauge_moves():
    pm = PromMetrics()
    with pm.inflight("/search"):
        assert 'bg_inflight_requests{endpoint="/search"} 1.0' in pm.render()[0].decode()
    assert 'bg_inflight_requests{endpoint="/search"} 0.0' in pm.render()[0].decode()


def test_two_instances_do_not_collide():
    PromMetrics()
    PromMetrics()  # global registry kullanılsaydı Duplicated timeseries hatası verirdi
```

- [ ] **Step 2: FAIL gör**
- [ ] **Step 3: Uygula** — `src/belge_gozu/telemetry/prom.py`:

```python
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from prometheus_client.core import CONTENT_TYPE_LATEST

from belge_gozu.telemetry.schema import RequestEvent

REQUEST_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30)
STAGE_BUCKETS = (0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20)
SCORE_BUCKETS = (45, 50, 55, 58, 60, 62, 65, 70, 75)
MARGIN_BUCKETS = (0, 0.5, 1, 2, 4, 8)
TPS_BUCKETS = (5, 10, 20, 40, 80, 160)

_STAGE_COLS = {
    "query_encode": "encode_ms",
    "stage1_hamming": "stage1_ms",
    "stage2_maxsim": "stage2_ms",
    "answerer": "answer_ms",
}


class PromMetrics:
    """Uygulama içi Prometheus kayıt defteri. Her örnek kendi registry'sini kurar."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("bg_http_requests", "İstek sayısı",
                                ["endpoint", "status"], registry=r)
        self.duration = Histogram("bg_request_duration_seconds", "Uçtan uca süre",
                                  ["endpoint"], buckets=REQUEST_BUCKETS, registry=r)
        self.stage = Histogram("bg_stage_duration_seconds", "Aşama süresi",
                               ["stage"], buckets=STAGE_BUCKETS, registry=r)
        self.top_score = Histogram("bg_retrieval_top_score", "En iyi skor",
                                   buckets=SCORE_BUCKETS, registry=r)
        self.margin = Histogram("bg_retrieval_score_margin", "top1-top2 farkı",
                                buckets=MARGIN_BUCKETS, registry=r)
        self.abstain = Counter("bg_abstain", "Abstain sayısı", ["reason"], registry=r)
        self.honest_miss = Counter("bg_honest_miss", "'bulamadım' yanıtları", registry=r)
        self.tokens = Counter("bg_llm_tokens", "LLM token sayısı", ["direction"], registry=r)
        self.tps = Histogram("bg_llm_tokens_per_second", "Üretim hızı",
                             buckets=TPS_BUCKETS, registry=r)
        self.cost = Counter("bg_llm_cost_usd", "Tahmini maliyet (USD)", registry=r)
        self.inflight_g = Gauge("bg_inflight_requests", "Anlık istek",
                                ["endpoint"], registry=r)
        self.pages = Gauge("bg_index_pages", "Dizindeki sayfa sayısı", registry=r)
        self.info = Info("bg_app", "Uygulama künyesi", registry=r)

    def set_app_info(self, *, pages: int, retriever_model: str, gemini_model: str,
                     device: str, version: str, threshold: float) -> None:
        self.pages.set(pages)
        self.info.info({"retriever_model": retriever_model, "gemini_model": gemini_model,
                        "device": device, "version": version, "threshold": str(threshold)})

    @contextmanager
    def inflight(self, endpoint: str) -> Iterator[None]:
        g = self.inflight_g.labels(endpoint=endpoint)
        g.inc()
        try:
            yield
        finally:
            g.dec()

    def observe(self, ev: RequestEvent) -> None:
        self.requests.labels(endpoint=ev.endpoint, status=ev.status).inc()
        self.duration.labels(endpoint=ev.endpoint).observe(ev.total_ms / 1000.0)
        for stage_name, col in _STAGE_COLS.items():
            v = getattr(ev, col)
            if v is not None:
                self.stage.labels(stage=stage_name).observe(v / 1000.0)
        if ev.top_score is not None:
            self.top_score.observe(ev.top_score)
        if ev.margin_1_2 is not None:
            self.margin.observe(ev.margin_1_2)
        if ev.status == "degraded":
            self.abstain.labels(reason="degraded").inc()
        elif ev.abstained:
            self.abstain.labels(reason="threshold").inc()
        if ev.honest_miss:
            self.honest_miss.inc()
        if ev.tokens_in:
            self.tokens.labels(direction="input").inc(ev.tokens_in)
        if ev.tokens_out:
            self.tokens.labels(direction="output").inc(ev.tokens_out)
        if ev.tokens_per_s is not None:
            self.tps.observe(ev.tokens_per_s)
        if ev.est_cost_usd:
            self.cost.inc(ev.est_cost_usd)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
```

Not: `process_*` kolektörleri app entegrasyonunda (Task 6) eklenir:
`ProcessCollector(registry=...)`, `PlatformCollector(registry=...)`, `GCCollector(registry=...)`.

- [ ] **Step 4: PASS gör**; `uv sync` sonrası `uv run pytest tests/telemetry/ -q`
- [ ] **Step 5: Tam süit + lint**
- [ ] **Step 6: Commit** — `git commit -m "feat(telemetry): Prometheus metrik katmanı (bg_* serileri)"`

---

### Task 5: Aşama enstrümantasyonu (retrieval + AskService)

**Files:**
- Modify: `src/belge_gozu/retrieval/core.py`, `src/belge_gozu/answer/base.py`
- Test: `tests/telemetry/test_stages_integration.py` (yeni)

**Interfaces:**
- Consumes: `stage`, `collecting`, `annotate` (Task 1)
- Produces: `/search` yolu `query_encode`, `stage1_hamming`, `stage2_maxsim` aşamalarını;
  `/ask` yolu ek olarak `answerer` aşamasını doldurur. `AskService.ask` degraded yolda
  `annotate("degraded", True)` çağırır. İmzalar DEĞİŞMEZ.

- [ ] **Step 1: Başarısız testi yaz** — `tests/telemetry/test_stages_integration.py`
(mevcut `tiny_corpus` fixture'ı `tests/conftest.py`'de; FakeEncoder'lı gerçek retriever kurar):

```python
import pandas as pd

from belge_gozu.answer.base import AskService
from belge_gozu.index.store import PackedIndex
from belge_gozu.retrieval.core import TwoStageRetriever
from belge_gozu.telemetry.collect import collecting


class StubAnswerer:
    def answer(self, question, pages, image_loader):
        from belge_gozu.answer.base import Answer
        return Answer(text="yanıt", citations=[pages[0].page_id])


def _retriever(tiny_corpus) -> TwoStageRetriever:
    data_dir, enc, _ = tiny_corpus
    index = PackedIndex.load(data_dir / "index")
    meta = pd.read_parquet(data_dir / "index" / "meta.parquet")
    return TwoStageRetriever(index, meta, enc)


def test_search_fills_retrieval_stages(tiny_corpus):
    r = _retriever(tiny_corpus)
    with collecting() as col:
        hits = r.search("deneme", k=3, candidates=10)
    assert hits
    assert {"query_encode", "stage1_hamming", "stage2_maxsim"} <= set(col.stages)


def test_ask_fills_answerer_stage(tiny_corpus):
    r = _retriever(tiny_corpus)
    svc = AskService(r, StubAnswerer(), min_score=-1e9, image_loader=lambda p: b"x")
    with collecting() as col:
        svc.ask("deneme", k=3, candidates=10)
    assert "answerer" in col.stages


def test_abstain_skips_answerer_stage(tiny_corpus):
    r = _retriever(tiny_corpus)
    svc = AskService(r, StubAnswerer(), min_score=1e9, image_loader=lambda p: b"x")
    with collecting() as col:
        answer, _ = svc.ask("deneme", k=3, candidates=10)
    assert answer.abstained and "answerer" not in col.stages
```

- [ ] **Step 2: FAIL gör**
- [ ] **Step 3: Uygula** — `retrieval/core.py` `search()`:

```python
    def search(self, query: str, k: int = 5, candidates: int = 200) -> list[PageHit]:
        if self.encoder is None:
            raise RuntimeError("encoder yapılandırılmamış")
        with stage("query_encode"):
            q_emb = self.encoder.encode_query(query)
        hits = self.search_embedding(q_emb, k, candidates)
        ...
```

`search_embedding()` içinde Aşama 1 bloğu `with stage("stage1_hamming"):`, Aşama 2
list-comprehension + sort bloğu `with stage("stage2_maxsim"):` ile sarılır
(import: `from belge_gozu.telemetry.collect import stage`).

`answer/base.py` `AskService.ask` — answerer çağrısı ve degraded yolu:

```python
        try:
            with stage("answerer"):
                return self.answerer.answer(question, hits, self.image_loader), hits
        except Exception:
            logger.exception("answerer failed")
            annotate("degraded", True)
            return Answer(text=SERVICE_ERROR_TEXT, citations=[], abstained=True), hits
```

- [ ] **Step 4: PASS gör** — `uv run pytest tests/telemetry/ tests/retrieval/ tests/answer/ -q`
- [ ] **Step 5: Tam süit + lint**
- [ ] **Step 6: Commit** — `git commit -m "feat(telemetry): retrieval + AskService aşama enstrümantasyonu"`

---

### Task 6: App entegrasyonu — olay birleştirme, /metrics, /stats

**Files:**
- Modify: `src/belge_gozu/app/main.py` (eski `_log_db/_log_write/log` KALDIRILIR), `src/belge_gozu/config.py` (`log_query_text: bool = True`)
- Test: `tests/app/test_api.py` (güncelleme + ekleme)

**Interfaces:**
- Consumes: Task 1–5'in tamamı
- Produces:
  - `GET /metrics` → Prometheus text format
  - `/stats` → `{"requests": int, "avg_ms": float, "p95_ms": float, "abstain_rate": float, "by_endpoint": {"/ask": int, "/search": int}}`
  - Her `/ask` ve `/search` isteği `events`'e bir satır yazar
  - `create_app`'e `recorder: EventRecorder | None = None` parametresi (test enjeksiyonu)

- [ ] **Step 1: Testleri güncelle/ekle** — `tests/app/test_api.py`'de:
  - `test_log_write_never_raises` SİLİNİR (halefi Task 2'de).
  - Eklenen testler:

```python
import sqlite3


def test_metrics_endpoint_exposes_series(tiny_corpus):
    c = make_client(tiny_corpus)
    c.post("/search", json={"query": "deneme"})
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "bg_http_requests_total" in r.text and "bg_stage_duration_seconds" in r.text


def test_events_row_written_for_ask(tiny_corpus):
    data_dir, _, _ = tiny_corpus
    c = make_client(tiny_corpus)
    c.post("/ask", json={"question": "kira artışı nedir?"})
    row = sqlite3.connect(data_dir / "requests.sqlite").execute(
        "SELECT endpoint, status, query_text, query_sha256, encode_ms, top_score "
        "FROM events WHERE endpoint='/ask'").fetchone()
    assert row[0] == "/ask" and row[1] == "answered"
    assert row[2] == "kira artışı nedir?" and len(row[3]) == 64
    assert row[4] is not None and row[5] is not None


def test_query_text_flag_off_hashes_only(tiny_corpus):
    data_dir, enc, _ = tiny_corpus
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index",
                        min_score_threshold=-1e9, log_query_text=False)
    app = create_app(settings=settings, encoder=enc, answerer=StubAnswerer())
    c = TestClient(app)
    c.post("/search", json={"query": "gizli soru"})
    row = sqlite3.connect(data_dir / "requests.sqlite").execute(
        "SELECT query_text, query_sha256 FROM events WHERE endpoint='/search' "
        "ORDER BY id DESC").fetchone()
    assert row[0] is None and len(row[1]) == 64


def test_stats_extended_shape(tiny_corpus):
    c = make_client(tiny_corpus)
    c.post("/ask", json={"question": "soru?"})
    s = c.get("/stats").json()
    assert s["requests"] >= 1 and s["avg_ms"] >= 0
    assert "p95_ms" in s and "abstain_rate" in s and "by_endpoint" in s
```

- [ ] **Step 2: FAIL gör**
- [ ] **Step 3: Uygula** — `main.py` yeniden düzenlemesi (bütün olarak):
  - `_log_db`, `_log_write`, `log()` fonksiyonları ve `sqlite3`/`threading` importlarının
    log'a özgü kullanımı kaldırılır.
  - `create_app(settings=None, encoder=None, answerer=None, recorder=None)`:

```python
    from belge_gozu.telemetry.prom import PromMetrics
    from belge_gozu.telemetry.recorder import EventRecorder

    rec = recorder or EventRecorder(s.data_dir / "requests.sqlite")
    prom = PromMetrics()
    try:
        from prometheus_client import GCCollector, PlatformCollector, ProcessCollector

        ProcessCollector(registry=prom.registry)
        PlatformCollector(registry=prom.registry)
        GCCollector(registry=prom.registry)
    except Exception:  # bazı platformlarda ProcessCollector yoktur; telemetri isteği düşürmez
        pass
    try:
        from importlib.metadata import version as pkg_version

        app_version = pkg_version("belge-gozu")
    except Exception:
        app_version = "0.0.0"
    prom.set_app_info(pages=len(index.page_ids), retriever_model=s.retriever_model,
                      gemini_model=s.gemini_model, device=s.device,
                      version=app_version, threshold=s.min_score_threshold)
```

  - Ortak olay kurucu (route'ların üstünde, closure):

```python
    def build_event(*, endpoint: str, status: str, http_status: int, total_ms: float,
                    col: StageCollector, query: str, hits: list[PageHit],
                    answer: Answer | None = None, error_type: str | None = None,
                    k: int | None = None, candidates: int | None = None) -> RequestEvent:
        top = hits[0].score if hits else None
        margin = (hits[0].score - hits[1].score) if len(hits) >= 2 else None
        tokens_in = col.notes.get("tokens_in")
        tokens_out = col.notes.get("tokens_out")
        answer_ms = col.stages.get("answerer")
        tps = None
        if isinstance(tokens_out, int) and answer_ms and answer_ms > 0:
            tps = tokens_out / (answer_ms / 1000.0)
        cost = None
        if isinstance(tokens_in, int) and isinstance(tokens_out, int):
            cost = (tokens_in / 1e6) * s.gemini_price_in_usd_per_1m + \
                   (tokens_out / 1e6) * s.gemini_price_out_usd_per_1m
        honest_miss = None
        if answer is not None and not answer.abstained:
            honest_miss = "bulamadım" in answer.text.lower()  # sezgisel (spec §5)
        return RequestEvent(
            ts=datetime.now(UTC).isoformat(), endpoint=endpoint, status=status,
            http_status=http_status, total_ms=total_ms,
            encode_ms=col.stages.get("query_encode"),
            stage1_ms=col.stages.get("stage1_hamming"),
            stage2_ms=col.stages.get("stage2_maxsim"),
            answer_ms=answer_ms, top_score=top, margin_1_2=margin,
            abstained=answer.abstained if answer else None,
            honest_miss=honest_miss, k=k, candidates=candidates,
            query_len=len(query),
            query_text=query if s.log_query_text else None,
            query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            answer_len=len(answer.text) if answer else None,
            citations_n=len(answer.citations) if answer else None,
            tokens_in=tokens_in if isinstance(tokens_in, int) else None,
            tokens_out=tokens_out if isinstance(tokens_out, int) else None,
            tokens_per_s=tps, est_cost_usd=cost, error_type=error_type,
            detail={"hits": [{"page_id": h.page_id, "score": h.score} for h in hits],
                    "threshold": s.min_score_threshold},
        )
```

  - `/search` route'u:

```python
    @app.post("/search")
    def search(body: SearchBody) -> dict[str, list[PageHit]]:
        t0 = time.perf_counter()
        with collecting() as col, prom.inflight("/search"):
            try:
                hits = retriever.search(body.query, k=body.k or s.top_k,
                                        candidates=s.stage1_candidates)
            except Exception as e:
                ev = build_event(endpoint="/search", status="error", http_status=500,
                                 total_ms=(time.perf_counter() - t0) * 1000, col=col,
                                 query=body.query, hits=[], error_type=type(e).__name__)
                rec.record(ev); prom.observe(ev)
                raise
            ev = build_event(endpoint="/search", status="ok", http_status=200,
                             total_ms=(time.perf_counter() - t0) * 1000, col=col,
                             query=body.query, hits=hits,
                             k=body.k or s.top_k, candidates=s.stage1_candidates)
            rec.record(ev); prom.observe(ev)
        return {"hits": hits}
```

  - `/ask` route'u aynı kalıpta; status belirleme:

```python
            answer, hits = service.ask(body.question, k=s.top_k, candidates=s.stage1_candidates)
            if col.notes.get("degraded"):
                status = "degraded"
            elif answer.abstained:
                status = "abstained"
            else:
                status = "answered"
```

  - Yeni endpoint'ler:

```python
    @app.get("/metrics")
    def metrics() -> Response:
        body, ctype = prom.render()
        return Response(content=body, media_type=ctype)

    @app.get("/stats")
    def stats() -> dict:
        db = sqlite3.connect(s.data_dir / "requests.sqlite")
        n, avg = db.execute("SELECT COUNT(*), COALESCE(AVG(total_ms),0) FROM events").fetchone()
        vals = [r[0] for r in db.execute(
            "SELECT total_ms FROM events ORDER BY id DESC LIMIT 10000")]
        vals.sort()
        p95 = vals[int(len(vals) * 0.95) - 1] if vals else 0.0
        ab = db.execute("SELECT COALESCE(AVG(abstained),0) FROM events "
                        "WHERE endpoint='/ask'").fetchone()[0]
        by = dict(db.execute("SELECT endpoint, COUNT(*) FROM events GROUP BY endpoint"))
        db.close()
        return {"requests": n, "avg_ms": round(avg, 1), "p95_ms": round(p95, 1),
                "abstain_rate": round(ab, 3), "by_endpoint": by}
```

  - Gerekli importlar: `hashlib`, `time`, `sqlite3`, `fastapi.responses.Response`,
    `belge_gozu.telemetry.collect.collecting/StageCollector`, `schema.RequestEvent`.
  - `config.py`: `log_query_text: bool = True` alanı eklenir.

- [ ] **Step 4: PASS gör** — `uv run pytest tests/app/ -q`
- [ ] **Step 5: Tam süit + lint**
- [ ] **Step 6: Commit** — `git commit -m "feat(telemetry): app entegrasyonu — events kaydı, /metrics, genişletilmiş /stats"`

---

### Task 7: Dışa aktarma + CLI (`export.py`, `metrics` komutları)

**Files:**
- Create: `src/belge_gozu/telemetry/export.py`
- Modify: `src/belge_gozu/cli.py`
- Test: `tests/telemetry/test_export.py`, `tests/test_cli.py` (ekleme)

**Interfaces:**
- Consumes: `events` tablosu (Task 2 DDL)
- Produces:
  - `export_events(db_path: Path, out: Path) -> int` — satır sayısı döner; `.parquet`/`.csv` uzantısına göre format
  - CLI: `belge-gozu metrics export --out PATH`, `belge-gozu metrics summary`

- [ ] **Step 1: Başarısız testleri yaz** — `tests/telemetry/test_export.py`:

```python
import pandas as pd

from belge_gozu.telemetry.export import export_events
from belge_gozu.telemetry.recorder import EventRecorder
from belge_gozu.telemetry.schema import RequestEvent


def test_export_parquet_roundtrip(tmp_path):
    rec = EventRecorder(tmp_path / "r.sqlite")
    rec.record(RequestEvent(ts="t1", endpoint="/search", status="ok", http_status=200,
                            total_ms=5.0, query_sha256="e" * 64))
    rec.close()
    out = tmp_path / "events.parquet"
    n = export_events(tmp_path / "r.sqlite", out)
    assert n == 1
    df = pd.read_parquet(out)
    assert list(df["endpoint"]) == ["/search"] and "total_ms" in df.columns


def test_export_csv(tmp_path):
    rec = EventRecorder(tmp_path / "r.sqlite")
    rec.record(RequestEvent(ts="t1", endpoint="/ask", status="answered", http_status=200,
                            total_ms=5.0, query_sha256="e" * 64))
    rec.close()
    n = export_events(tmp_path / "r.sqlite", tmp_path / "events.csv")
    assert n == 1 and (tmp_path / "events.csv").read_text().startswith("id,")
```

`tests/test_cli.py`'ye ekleme (mevcut CliRunner kalıbını izle; dosyayı önce oku):

```python
def test_metrics_export_cli(tmp_path, monkeypatch):
    from belge_gozu.telemetry.recorder import EventRecorder
    from belge_gozu.telemetry.schema import RequestEvent

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    rec = EventRecorder(tmp_path / "requests.sqlite")
    rec.record(RequestEvent(ts="t", endpoint="/search", status="ok", http_status=200,
                            total_ms=1.0, query_sha256="f" * 64))
    rec.close()
    result = runner.invoke(app, ["metrics", "export", "--out", str(tmp_path / "e.parquet")])
    assert result.exit_code == 0 and (tmp_path / "e.parquet").exists()
```

- [ ] **Step 2: FAIL gör**
- [ ] **Step 3: Uygula** — `export.py`:

```python
import sqlite3
from pathlib import Path

import pandas as pd


def export_events(db_path: Path, out: Path) -> int:
    """events tablosunu Parquet/CSV'ye döker; satır sayısını döner."""
    db = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM events ORDER BY id", db)
    finally:
        db.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    return len(df)
```

`cli.py`'ye `metrics_app = typer.Typer()` + `app.add_typer(metrics_app, name="metrics")` ve:

```python
@metrics_app.command("export")
def metrics_export(out: Path = typer.Option(Path("data/exports/events.parquet"))) -> None:  # noqa: B008
    from belge_gozu.telemetry.export import export_events

    s = _settings()
    n = export_events(s.data_dir / "requests.sqlite", out)
    typer.echo(f"{n} olay -> {out}")


@metrics_app.command("summary")
def metrics_summary() -> None:
    import sqlite3

    s = _settings()
    db = sqlite3.connect(s.data_dir / "requests.sqlite")
    n, avg = db.execute("SELECT COUNT(*), COALESCE(AVG(total_ms),0) FROM events").fetchone()
    ab = db.execute(
        "SELECT COALESCE(AVG(abstained),0) FROM events WHERE endpoint='/ask'").fetchone()[0]
    tok = db.execute("SELECT COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), "
                     "COALESCE(SUM(est_cost_usd),0) FROM events").fetchone()
    vals = sorted(r[0] for r in db.execute("SELECT total_ms FROM events"))
    p95 = vals[int(len(vals) * 0.95) - 1] if vals else 0.0
    db.close()
    typer.echo(f"istek={n} ort={avg:.0f}ms p95={p95:.0f}ms abstain={ab:.1%}")
    typer.echo(f"token in/out={tok[0]}/{tok[1]} maliyet≈${tok[2]:.4f}")
```

- [ ] **Step 4: PASS gör**
- [ ] **Step 5: Tam süit + lint**
- [ ] **Step 6: Commit** — `git commit -m "feat(telemetry): metrics export/summary CLI"`

---

### Task 8: Yük üreticisi (`scripts/loadgen.py`)

**Files:**
- Create: `scripts/loadgen.py`, `scripts/queries_sample.txt`
- Test: `tests/test_loadgen.py`

**Interfaces:**
- Consumes: çalışan sunucu (yalnız canlı kullanımda; test saf fonksiyonu hedefler)
- Produces: `summarize(latencies_ms: list[float], errors: int, duration_s: float) -> dict`
  (anahtarlar: `requests`, `errors`, `rps`, `p50_ms`, `p95_ms`, `p99_ms`)

- [ ] **Step 1: Başarısız testi yaz** — `tests/test_loadgen.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from loadgen import summarize  # noqa: E402


def test_summarize_percentiles():
    lat = [float(i) for i in range(1, 101)]  # 1..100 ms
    s = summarize(lat, errors=2, duration_s=10.0)
    assert s["requests"] == 100 and s["errors"] == 2
    assert s["rps"] == 10.0
    assert s["p50_ms"] == 50.0 and s["p95_ms"] == 95.0 and s["p99_ms"] == 99.0


def test_summarize_empty():
    s = summarize([], errors=0, duration_s=1.0)
    assert s["requests"] == 0 and s["p95_ms"] == 0.0
```

- [ ] **Step 2: FAIL gör**
- [ ] **Step 3: Uygula** — `scripts/loadgen.py`:

```python
"""Belge-Gözü yük üreticisi.

Varsayılan /search'tür: /ask Gemini kotası yakar (≈20 çağrı/gün) ve ancak
--endpoint ask --yes-burn-quota ile açılır. Örnek:
    uv run python scripts/loadgen.py --concurrency 8 --duration 60 --out out.json
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

QUERIES = Path(__file__).with_name("queries_sample.txt")


def summarize(latencies_ms: list[float], errors: int, duration_s: float) -> dict:
    lat = sorted(latencies_ms)

    def pct(p: float) -> float:
        if not lat:
            return 0.0
        return lat[min(len(lat) - 1, max(0, int(round(p * len(lat))) - 1))]

    return {
        "requests": len(lat),
        "errors": errors,
        "rps": round(len(lat) / duration_s, 2) if duration_s > 0 else 0.0,
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
    }


async def worker(client: httpx.AsyncClient, endpoint: str, questions: list[str],
                 stop_at: float, lat: list[float], errs: list[int]) -> None:
    while time.monotonic() < stop_at:
        q = random.choice(questions)
        body = {"question": q} if endpoint == "/ask" else {"query": q}
        t0 = time.perf_counter()
        try:
            r = await client.post(endpoint, json=body, timeout=120)
            if r.status_code == 200:
                lat.append((time.perf_counter() - t0) * 1000)
            else:
                errs[0] += 1
        except Exception:
            errs[0] += 1


async def run(args: argparse.Namespace) -> dict:
    questions = [q.strip() for q in QUERIES.read_text().splitlines() if q.strip()]
    endpoint = "/ask" if args.endpoint == "ask" else "/search"
    lat: list[float] = []
    errs = [0]
    stop_at = time.monotonic() + args.duration
    async with httpx.AsyncClient(base_url=args.base_url) as client:
        t0 = time.monotonic()
        await asyncio.gather(*(worker(client, endpoint, questions, stop_at, lat, errs)
                               for _ in range(args.concurrency)))
        dur = time.monotonic() - t0
    return {"config": vars(args), "endpoint": endpoint, **summarize(lat, errs[0], dur)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:7860")
    ap.add_argument("--endpoint", choices=["search", "ask"], default="search")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--yes-burn-quota", action="store_true")
    args = ap.parse_args()
    if args.endpoint == "ask" and not args.yes_burn_quota:
        ap.error("/ask Gemini kotası yakar; bilinçliysen --yes-burn-quota ekle")
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

`scripts/queries_sample.txt` — 30 satır, birebir şu içerik:

```
Kişisel Verilerin Korunması Kanunu'na göre açık rızanın geçerlilik şartları nelerdir?
İş Kanunu'na göre yıllık ücretli izin süresi ne kadardır?
Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?
Türk Borçlar Kanunu'na göre kira artışı en fazla ne kadar olabilir?
Türk Ceza Kanunu'na göre kasten yaralama suçunun cezası nedir?
Anayasa'ya göre temel hak ve hürriyetler hangi hallerde sınırlanabilir?
Türk Ticaret Kanunu'na göre anonim şirket asgari sermayesi ne kadardır?
Vergi Usul Kanunu'na göre defter tutma yükümlülüğü kimlere aittir?
Katma Değer Vergisi Kanunu'na göre KDV oranını belirlemeye kim yetkilidir?
İş Sağlığı ve Güvenliği Kanunu'na göre işverenin genel yükümlülükleri nelerdir?
Tüketicinin Korunması Hakkında Kanun'a göre cayma hakkı süresi kaç gündür?
İcra ve İflas Kanunu'na göre haciz nasıl uygulanır?
Sosyal Sigortalar Kanunu'na göre emeklilik yaşı nasıl belirlenir?
Kamulaştırma Kanunu'na göre kamulaştırma bedeli nasıl tespit edilir?
Rekabetin Korunması Hakkında Kanun'a göre hakim durumun kötüye kullanılması nedir?
Çevre Kanunu'na göre kirleten öder ilkesi ne anlama gelir?
Kişisel Verilerin Korunması Kanunu'na göre veri sorumlusunun yükümlülükleri nelerdir?
İş Kanunu'na göre kıdem tazminatı hangi hallerde ödenir?
Türk Medeni Kanunu'na göre evlenme yaşı kaçtır?
Türk Borçlar Kanunu'na göre zamanaşımı süresi genel olarak kaç yıldır?
Anayasa'ya göre Cumhurbaşkanının görev süresi kaç yıldır?
Türk Ceza Kanunu'na göre hırsızlık suçunun temel cezası nedir?
Türk Ticaret Kanunu'na göre limited şirket kaç ortakla kurulabilir?
Gelir Vergisi Kanunu'na göre gelir vergisine tabi kazançlar nelerdir?
İş Kanunu'na göre haftalık çalışma süresi en fazla kaç saattir?
Tüketicinin Korunması Hakkında Kanun'a göre ayıplı mal nedir?
Sendikalar ve Toplu İş Sözleşmesi Kanunu'na göre grev hakkı nasıl kullanılır?
Türk Medeni Kanunu'na göre mirasçılık belgesi nereden alınır?
Kişisel Verilerin Korunması Kanunu'na göre veri ihlali bildirimi kaç gün içinde yapılır?
Anayasa'ya göre yasama yetkisi kime aittir?
```

- [ ] **Step 4: PASS gör** — `uv run pytest tests/test_loadgen.py -q`
- [ ] **Step 5: Tam süit + lint** (ruff `scripts/`i de tarar; `src` ayarı nedeniyle taramıyorsa `pyproject.toml`'a dokunma — mevcut davranışı koru)
- [ ] **Step 6: Commit** — `git commit -m "feat(telemetry): loadgen + örnek sorgu seti"`

---

### Task 9: Observability stack (compose + Grafana) + docs/research iskeleti

**Files:**
- Create: `observability/docker-compose.yml`, `observability/prometheus.yml`,
  `observability/grafana/provisioning/datasources/prometheus.yml`,
  `observability/grafana/provisioning/dashboards/provider.yml`,
  `observability/grafana/provisioning/dashboards/belge-gozu.json`,
  `docs/research/metrics-catalog.md`, `docs/research/runbook.md`
- Modify: `Makefile` (`obs-up`, `obs-down` hedefleri)

**Interfaces:**
- Consumes: `GET /metrics` (Task 6)
- Produces: `make obs-up` → Prometheus :9090 + Grafana :3000 (anonim erişim, provisioned dashboard "Belge-Gözü")

- [ ] **Step 1: Compose + Prometheus config**

`observability/docker-compose.yml`:

```yaml
services:
  prometheus:
    image: prom/prometheus:v2.53.0
    ports: ["9090:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prom-data:/prometheus
    extra_hosts: ["host.docker.internal:host-gateway"]
  grafana:
    image: grafana/grafana:11.1.0
    ports: ["3000:3000"]
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: "Admin"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - graf-data:/var/lib/grafana
volumes:
  prom-data:
  graf-data:
```

`observability/prometheus.yml`:

```yaml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: belge-gozu
    static_configs:
      - targets: ["host.docker.internal:7860"]
```

`observability/grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

`observability/grafana/provisioning/dashboards/provider.yml`:

```yaml
apiVersion: 1
providers:
  - name: belge-gozu
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

- [ ] **Step 2: Dashboard JSON** — `observability/grafana/provisioning/dashboards/belge-gozu.json`.
9 panel; her panel `datasource: {"type":"prometheus","uid":"prometheus"}` kullanır.
Panel başlıkları ve PromQL ifadeleri birebir:

| Panel (type) | expr |
|---|---|
| RPS (timeseries) | `sum by (endpoint) (rate(bg_http_requests_total[1m]))` |
| Gecikme p50/p95/p99 (timeseries) | `histogram_quantile(0.5, sum by (le,endpoint) (rate(bg_request_duration_seconds_bucket[5m])))` + 0.95 + 0.99 varyantları |
| Aşama süreleri (timeseries, stacked) | `histogram_quantile(0.95, sum by (le,stage) (rate(bg_stage_duration_seconds_bucket[5m])))` |
| Abstain oranı % (stat) | `100 * sum(rate(bg_abstain_total[15m])) / sum(rate(bg_http_requests_total{endpoint="/ask"}[15m]))` |
| Top skor dağılımı (heatmap) | `sum by (le) (rate(bg_retrieval_top_score_bucket[5m]))` (format: heatmap) |
| Token/sn p50 (stat) | `histogram_quantile(0.5, sum by (le) (rate(bg_llm_tokens_per_second_bucket[15m])))` |
| Toplam token (timeseries) | `sum by (direction) (increase(bg_llm_tokens_total[1h]))` |
| Kümülatif maliyet USD (stat) | `bg_llm_cost_usd_total` |
| Inflight + RSS (timeseries) | `bg_inflight_requests` ve `process_resident_memory_bytes` |

JSON iskeleti (panel dizisi yukarıdaki tabloyu uygular; her panel `gridPos` ile 2 sütunlu
yerleşime oturtulur, `schemaVersion: 39`, `title: "Belge-Gözü"`, `uid: "belge-gozu"`,
`refresh: "5s"`). Uygulayıcı: tabloyu JSON'a birebir çevir; başka panel ekleme.

- [ ] **Step 3: Makefile hedefleri** (mevcut Makefile'a ekle):

```makefile
obs-up:
	docker compose -f observability/docker-compose.yml up -d
obs-down:
	docker compose -f observability/docker-compose.yml down
```

- [ ] **Step 4: docs/research iskeleti**

`docs/research/metrics-catalog.md` — spec §5 (olay kolonları) ve §6 (Prometheus
kataloğu + bucket'lar + türetilmiş metrikler) tablolarını insan-okur formatta içerir;
her metrik için: ad, tip, birim, etiketler, kaynak kod noktası, "neden önemli", faz.
Spec'ten kopyalanır ve `honest_miss`'in sezgisel olduğu açıkça işaretlenir.

`docs/research/runbook.md` — birebir bölümler:

```markdown
# Telemetri Runbook

## Stack'i kaldır
1. Sunucu: `uv run belge-gozu serve` (host'ta, :7860)
2. `make obs-up` → Prometheus http://localhost:9090 · Grafana http://localhost:3000
3. Doğrula: Prometheus Targets sayfasında `belge-gozu` UP; Grafana'da "Belge-Gözü" dashboard'u.

## Ölçüm oturumu koş
1. `uv run python scripts/loadgen.py --concurrency 8 --duration 60 --out docs/research/findings/raw/$(date +%F)-loadgen.json`
2. Gerçek yanıt yolu için EN FAZLA 2-3 `/ask` sorusu (kota: ≈20/gün): UI'dan ya da curl ile.
3. `uv run belge-gozu metrics summary` çıktısını not al.
4. `uv run belge-gozu metrics export --out data/exports/$(date +%F)-events.parquet`

## Bulgu notu yaz
`docs/research/findings/YYYY-MM-DD-<konu>.md`: koşum künyesi (commit sha, config,
korpus boyutu, yük parametreleri) + gözlemler + figür referansları. Grafana panel
görüntüleri `docs/research/figures/` altına PNG olarak.

## Fiyat varsayımını doğrula
`BG_GEMINI_PRICE_IN_USD_PER_1M` / `BG_GEMINI_PRICE_OUT_USD_PER_1M` varsayılanları
tahminidir. Güncel Gemini Flash fiyatını resmî fiyat sayfasından kontrol et; farklıysa
`.env`'e yaz ve bulgu notunda belirt.

## Kota bütçesi
Gemini ≈20 çağrı/gün. Loadgen ASLA `/ask` ile koşulmaz (bayrak korumalı). CI hiç
çağrı yapmaz.
```

- [ ] **Step 5: Compose'u canlı doğrula** — `make obs-up`; `curl -s localhost:9090/api/v1/targets | grep -o '"health":"[a-z]*"'` → `"health":"up"` (sunucu :7860'ta koşuyorken); `curl -s localhost:3000/api/health` → `"database": "ok"`. Sorun çıkarsa düzelt (OrbStack'te `host.docker.internal` `host-gateway` ile çalışır).
- [ ] **Step 6: Commit** — `git add observability docs/research Makefile && git commit -m "feat(telemetry): Prometheus+Grafana compose, provisioned dashboard, runbook + katalog"`

---

### Task 10: Canlı doğrulama + baseline bulgu notu + README

**Files:**
- Create: `docs/research/findings/2026-08-26-baseline.md`, `docs/research/figures/` (PNG'ler), `docs/research/findings/raw/` (loadgen JSON)
- Modify: `README.md` (Telemetri bölümü)

**Interfaces:**
- Consumes: Task 1–9'un tamamı, çalışan sunucu, OrbStack

- [ ] **Step 1: Sunucuyu yeni kodla yeniden başlat**, `/healthz` bekle; `make obs-up`.
- [ ] **Step 2: Loadgen koş** — `uv run python scripts/loadgen.py --concurrency 8 --duration 60 --out docs/research/findings/raw/2026-08-26-loadgen.json` (yalnız `/search`; kota yanmaz).
- [ ] **Step 3: 2 gerçek `/ask`** — `queries_sample.txt`'ten 1 cevap-beklenen (KVKK açık rıza) + 1 abstain-beklenen (TMK yerleşim yeri) soru; UI ya da curl.
- [ ] **Step 4: Kanıt topla** — `belge-gozu metrics summary` çıktısı; `metrics export` parquet'i; `curl -s localhost:7860/metrics | grep -c '^bg_'` sayısı; Grafana dashboard'unun dolu hali (`docs/research/figures/2026-08-26-dashboard.png` — Playwright ile `http://localhost:3000/d/belge-gozu` ekran görüntüsü alınabilir).
- [ ] **Step 5: Baseline bulgu notunu GERÇEK sayılarla yaz** — şablon:

```markdown
# 2026-08-26 — Telemetri baseline ölçümü

## Koşum künyesi
- commit: <sha> · korpus: 4222 sayfa · device: <mps|cpu> · threshold: 60.0
- yük: /search, concurrency=8, 60 sn

## Sonuçlar (loadgen, istemci tarafı)
- throughput: <X> rps · p50=<X> ms · p95=<X> ms · p99=<X> ms · hata=<X>

## Sunucu tarafı (metrics summary + Grafana)
- aşama kırılımı p95: encode=<X> ms · stage1=<X> ms · stage2=<X> ms
- /ask örnekleri: total=<X> ms · answerer=<X> ms · tokens in/out=<X>/<X>
  · token/sn=<X> · maliyet≈$<X>
- abstain oranı (bu oturum): <X>

## Gözlemler
- <encode mi darboğaz? eşzamanlılıkta ne oldu? beklenmedik bir şey?>

## Figürler
- figures/2026-08-26-dashboard.png
```

- [ ] **Step 6: README'ye "Telemetri" bölümü** — Quickstart'ın altına ~10 satır: neyin
ölçüldüğü (aşama kırılımı, token/maliyet, abstain), `/metrics`, `make obs-up`,
`metrics summary/export`, `docs/research/` işaretçisi.
- [ ] **Step 7: Tam süit + lint son kontrol**; `make obs-down` (isteğe bağlı, açık da kalabilir).
- [ ] **Step 8: Commit** — `git commit -m "docs(telemetry): baseline bulgu notu + README telemetri bölümü"`

---

## Self-Review Sonucu

- Spec kapsaması: §4→T1-2-4-7, §5→T1-2, §6→T4, §7→T3-5-6, §8→T9, §9→T8, §10→T7, §11→T9-10, §12→tüm task testleri, §14 kabul ölçütleri→T6 (1,2*), T3+T10 (3), T7 (4), tümü (5), T10 (6). Boşluk yok.
- Tip tutarlılığı: `GenResult`, `RequestEvent`, `StageCollector`, `EventRecorder.record`, `PromMetrics.observe/render/inflight`, `export_events`, `summarize` imzaları task'lar arasında birebir aynı.
- Yer tutucu taraması: kod blokları eksiksiz; Task 6'daki `/ask` route'u Task 6 Step 3'teki kalıbı `/search` örneğinden uygular (status belirleme bloğu verildi); Task 9 dashboard JSON'u tablodan üretilir (expr'ler verbatim verildi).
