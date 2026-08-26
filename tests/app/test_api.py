import threading

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


def test_log_write_never_raises():
    import sqlite3 as sq
    from unittest.mock import MagicMock

    from belge_gozu.app.main import _log_write

    bad_db = MagicMock()
    bad_db.execute.side_effect = sq.OperationalError("database is locked")
    _log_write(bad_db, threading.Lock(), "/search", 1.0, 0.5)  # ne exception ne 500
