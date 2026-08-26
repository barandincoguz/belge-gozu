import sqlite3

from fastapi.testclient import TestClient

from belge_gozu.answer.base import Answer
from belge_gozu.app.main import create_app
from belge_gozu.config import Settings


class StubAnswerer:
    def answer(self, question, pages, image_loader):
        return Answer(text=f"yanıt: {question}", citations=[pages[0].page_id])


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


def test_events_row_written_for_ask(tiny_corpus):
    data_dir, _, _ = tiny_corpus
    c = make_client(tiny_corpus)
    c.post("/ask", json={"question": "kira artışı nedir?"})
    row = (
        sqlite3.connect(data_dir / "requests.sqlite")
        .execute(
            "SELECT endpoint, status, query_text, query_sha256, encode_ms, top_score "
            "FROM events WHERE endpoint='/ask'"
        )
        .fetchone()
    )
    assert row[0] == "/ask" and row[1] == "answered"
    assert row[2] == "kira artışı nedir?" and len(row[3]) == 64
    assert row[4] is not None and row[5] is not None


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
