import json
import sqlite3

from fastapi.testclient import TestClient

from belge_gozu.answer.base import SERVICE_ERROR_TEXT, Answer
from belge_gozu.app.main import create_app
from belge_gozu.config import Settings
from belge_gozu.telemetry.recorder import EventRecorder


class StubAnswerer:
    def answer(self, question, pages, image_loader):
        return Answer(text=f"yanıt: {question}", citations=[pages[0].page_id])


class BoomAnswerer:
    """answer() her koşulda patlar — AskService'in degrade-guard'ını tetikler."""

    def answer(self, question, pages, image_loader):
        raise RuntimeError("boom")


class BoomEncoder:
    """encode_query() her koşulda patlar — retriever.search()'ün başarısız olduğu yol."""

    def encode_pages(self, images):
        raise NotImplementedError

    def encode_query(self, text):
        raise RuntimeError("boom")


class BoomRecorder(EventRecorder):
    """record() her koşulda patlar — telemetri hatasının isteği düşürmediğini kilitler."""

    def record(self, ev):
        raise RuntimeError("boom")


def make_client(tiny_corpus) -> TestClient:
    data_dir, enc, _ = tiny_corpus
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index", min_score_threshold=-1e9)
    app = create_app(settings=settings, encoder=enc, answerer=StubAnswerer())
    return TestClient(app)


def test_healthz(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "pages": 3, "threshold": -1e9}


def test_search_returns_hits(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.post("/search", json={"query": "deneme sorgusu"})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 3 and {"page_id", "score", "image_path"} <= hits[0].keys()


def test_ask_returns_answer_and_logs(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.post("/ask", json={"question": "kira artışı nedir?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"]["text"].startswith("yanıt:") and body["answer"]["citations"]
    stats = c.get("/stats").json()
    assert stats["requests"] >= 1 and stats["avg_ms"] >= 0


def test_page_image_served(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.get("/pages/images/d0/0001.webp")
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"


def test_root_serves_ui(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.get("/")
    assert r.status_code == 200
    assert "Belge-Gözü" in r.text
    assert 'id="q"' in r.text and 'id="ask-btn"' in r.text  # gerçek UI yüklendi


def test_metrics_endpoint_exposes_series(tiny_corpus):
    c = make_client(tiny_corpus)
    c.post("/search", json={"query": "deneme"})
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "bg_http_requests_total" in r.text and "bg_stage_duration_seconds" in r.text
    assert 'bg_http_requests_total{endpoint="/search",status="ok"} 1.0' in r.text


def test_events_row_written_for_ask(tiny_corpus):
    data_dir, _, _ = tiny_corpus
    c = make_client(tiny_corpus)
    c.post("/ask", json={"question": "kira artışı nedir?"})
    row = (
        sqlite3.connect(data_dir / "requests.sqlite")
        .execute(
            "SELECT endpoint, status, query_text, query_sha256, encode_ms, top_score, detail "
            "FROM events WHERE endpoint='/ask'"
        )
        .fetchone()
    )
    assert row[0] == "/ask" and row[1] == "answered"
    assert row[2] == "kira artışı nedir?" and len(row[3]) == 64
    assert row[4] is not None and row[5] is not None
    # detail: koşum künyesi (item 3) — hits/threshold korunur, model/device/version eklenir.
    detail = json.loads(row[6])
    assert "hits" in detail and "threshold" in detail
    assert detail["retriever_model"] and detail["gemini_model"]
    assert "device" in detail and detail["app_version"]


def test_search_detail_records_exhaustive_stage_timing(tiny_corpus):
    """Varsayılan pipeline ("exhaustive"): exhaustive_maxsim süresi detail.stages'e düşer."""
    data_dir, _, _ = tiny_corpus
    c = make_client(tiny_corpus)
    c.post("/search", json={"query": "deneme sorgusu"})
    row = (
        sqlite3.connect(data_dir / "requests.sqlite")
        .execute("SELECT detail FROM events WHERE endpoint='/search' ORDER BY id DESC")
        .fetchone()
    )
    detail = json.loads(row[0])
    assert "stages" in detail
    assert "exhaustive_maxsim" in detail["stages"]


def test_search_records_pipeline_and_index_revision(tiny_corpus):
    """T13: pipeline + index_revision doldurulur; detail.retrieval kimlik alanlarını taşır."""
    data_dir, _, _ = tiny_corpus
    c = make_client(tiny_corpus)
    c.post("/search", json={"query": "deneme sorgusu"})
    row = (
        sqlite3.connect(data_dir / "requests.sqlite")
        .execute(
            "SELECT pipeline, index_revision, detail FROM events "
            "WHERE endpoint='/search' ORDER BY id DESC"
        )
        .fetchone()
    )
    assert row[0] == "exhaustive"
    assert row[1] is not None and "train-compat-v1" in row[1]
    detail = json.loads(row[2])
    assert detail["retrieval"] == {"query_format": "train-compat-v1", "quantization": "sign-1bit"}
    assert "candidates" not in detail["retrieval"]


def test_query_text_flag_off_hashes_only(tiny_corpus):
    data_dir, enc, _ = tiny_corpus
    settings = Settings(
        data_dir=data_dir,
        index_dir=data_dir / "index",
        min_score_threshold=-1e9,
        log_query_text=False,
    )
    app = create_app(settings=settings, encoder=enc, answerer=StubAnswerer())
    c = TestClient(app)
    c.post("/search", json={"query": "gizli soru"})
    row = (
        sqlite3.connect(data_dir / "requests.sqlite")
        .execute(
            "SELECT query_text, query_sha256 FROM events WHERE endpoint='/search' ORDER BY id DESC"
        )
        .fetchone()
    )
    assert row[0] is None and len(row[1]) == 64


def test_stats_extended_shape(tiny_corpus):
    c = make_client(tiny_corpus)
    c.post("/ask", json={"question": "soru?"})
    s = c.get("/stats").json()
    assert s["requests"] >= 1 and s["avg_ms"] >= 0
    assert "p95_ms" in s and "abstain_rate" in s and "by_endpoint" in s


def test_ask_degrades_on_answerer_failure(tiny_corpus):
    data_dir, enc, _ = tiny_corpus
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index", min_score_threshold=-1e9)
    app = create_app(settings=settings, encoder=enc, answerer=BoomAnswerer())
    c = TestClient(app)
    r = c.post("/ask", json={"question": "soru?"})
    assert r.status_code == 200
    assert r.json()["answer"]["text"] == SERVICE_ERROR_TEXT
    row = (
        sqlite3.connect(data_dir / "requests.sqlite")
        .execute("SELECT status FROM events WHERE endpoint='/ask'")
        .fetchone()
    )
    assert row[0] == "degraded"
    # abstain_rate degraded satırları hariç tutmalı (bir /ask isteği ve o
    # degraded olduğu için abstain_rate 0.0 olmalı, 1.0 değil).
    stats = c.get("/stats").json()
    assert stats["abstain_rate"] == 0.0


def test_search_encoder_failure_500s_and_records_error(tiny_corpus):
    data_dir, _, _ = tiny_corpus
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index", min_score_threshold=-1e9)
    app = create_app(settings=settings, encoder=BoomEncoder(), answerer=StubAnswerer())
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/search", json={"query": "deneme"})
    assert r.status_code == 500
    rows = (
        sqlite3.connect(data_dir / "requests.sqlite")
        .execute("SELECT status, error_type FROM events WHERE endpoint='/search'")
        .fetchall()
    )
    assert len(rows) == 1
    assert rows[0][0] == "error" and rows[0][1] == "RuntimeError"


def test_search_survives_recorder_failure(tiny_corpus):
    data_dir, enc, _ = tiny_corpus
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index", min_score_threshold=-1e9)
    boom_rec = BoomRecorder(data_dir / "requests.sqlite")
    app = create_app(settings=settings, encoder=enc, answerer=StubAnswerer(), recorder=boom_rec)
    c = TestClient(app)
    r = c.post("/search", json={"query": "deneme"})
    assert r.status_code == 200  # olay kaydı patladı ama istek etkilenmedi
