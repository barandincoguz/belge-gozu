import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from belge_gozu.answer.base import AskService
from belge_gozu.config import Settings, get_settings
from belge_gozu.index.store import PackedIndex
from belge_gozu.retrieval.core import TwoStageRetriever
from belge_gozu.retrieval.types import PageHit

STATIC_DIR = Path(__file__).parent / "static"


class SearchBody(BaseModel):
    query: str
    k: int | None = None


class AskBody(BaseModel):
    question: str


def _log_db(settings: Settings) -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(settings.data_dir / "requests.sqlite", check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS log (ts TEXT, path TEXT, ms REAL, top_score REAL)")
    return db


def create_app(settings: Settings | None = None, encoder=None, answerer=None) -> FastAPI:
    s = settings or get_settings()
    index = PackedIndex.load(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    if encoder is None:
        from belge_gozu.index.encode import ColSmolEncoder

        encoder = ColSmolEncoder(s.retriever_model, s.device)
    if answerer is None:
        from belge_gozu.answer.gemini import GeminiAnswerer

        answerer = GeminiAnswerer(s.gemini_model, s.gemini_api_key)

    def load_image(image_path: str) -> bytes:
        return (s.data_dir / image_path).read_bytes()

    retriever = TwoStageRetriever(index, meta, encoder)
    service = AskService(retriever, answerer, s.min_score_threshold, load_image)
    db = _log_db(s)
    app = FastAPI(title="Belge-Gözü")

    def log(path: str, ms: float, top_score: float) -> None:
        db.execute(
            "INSERT INTO log VALUES (?,?,?,?)",
            (datetime.now(UTC).isoformat(), path, ms, top_score),
        )
        db.commit()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "pages": len(index.page_ids)}

    @app.post("/search")
    def search(body: SearchBody) -> dict[str, list[PageHit]]:
        t0 = time.perf_counter()
        hits = retriever.search(body.query, k=body.k or s.top_k, candidates=s.stage1_candidates)
        log("/search", (time.perf_counter() - t0) * 1000, hits[0].score if hits else 0.0)
        return {"hits": hits}

    @app.post("/ask")
    def ask(body: AskBody) -> dict:
        t0 = time.perf_counter()
        answer, hits = service.ask(body.question, k=s.top_k, candidates=s.stage1_candidates)
        log("/ask", (time.perf_counter() - t0) * 1000, hits[0].score if hits else 0.0)
        return {"answer": answer.model_dump(), "hits": [h.model_dump() for h in hits]}

    @app.get("/stats")
    def stats() -> dict:
        row = db.execute("SELECT COUNT(*), COALESCE(AVG(ms),0) FROM log").fetchone()
        return {"requests": row[0], "avg_ms": round(row[1], 1)}

    @app.get("/pages/{image_path:path}")
    def page_image(image_path: str) -> FileResponse:
        full = (s.data_dir / image_path).resolve()
        if not full.is_relative_to(s.data_dir.resolve()) or not full.exists():
            raise HTTPException(404)
        return FileResponse(full, media_type="image/webp")

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        page = STATIC_DIR / "index.html"
        if page.exists():
            return page.read_text(encoding="utf-8")
        return "<html><body><h1>Belge-Gözü</h1><p>UI yakında.</p></body></html>"

    return app
