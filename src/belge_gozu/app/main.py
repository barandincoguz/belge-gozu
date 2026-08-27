import hashlib
import logging
import math
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from belge_gozu.answer.base import Answer, AskService
from belge_gozu.config import Settings, get_settings
from belge_gozu.index.compat import IndexCompatibilityError, check_compatibility
from belge_gozu.index.manifest import (
    CPE_0_3_18,
    DOC_PROMPTS,
    QUERY_FORMATS,
    DocPromptChoice,
    QueryFormatChoice,
)
from belge_gozu.index.store import PackedIndex
from belge_gozu.retrieval.core import ExhaustiveBinaryRetriever, TwoStageRetriever
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import StageCollector, collecting
from belge_gozu.telemetry.prom import PromMetrics
from belge_gozu.telemetry.recorder import EventRecorder
from belge_gozu.telemetry.schema import RequestEvent

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class SearchBody(BaseModel):
    query: str
    k: int | None = None


class AskBody(BaseModel):
    question: str


def create_app(
    settings: Settings | None = None,
    encoder=None,
    answerer=None,
    recorder: EventRecorder | None = None,
) -> FastAPI:
    s = settings or get_settings()
    index = PackedIndex.load(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    # CLI'nin index build sırasında kullandığı QUERY_FORMATS/DOC_PROMPTS
    # sözlükleriyle aynı kaynak (belge_gozu.index.manifest) — serve config'i
    # (Settings.query_format_id/doc_prompt_id) buradan çözülür (T11/Step 6).
    resolved_query_format = QUERY_FORMATS[QueryFormatChoice(s.query_format_id)]
    resolved_doc_prompt = DOC_PROMPTS[DocPromptChoice(s.doc_prompt_id)]
    if encoder is None:
        from belge_gozu.index.encode import ColSmolEncoder

        encoder = ColSmolEncoder(
            s.retriever_model,
            s.device,
            query_format=resolved_query_format,
            visual_prompt_override=resolved_doc_prompt,
        )
    if answerer is None:
        from belge_gozu.answer.gemini import GeminiAnswerer

        answerer = GeminiAnswerer(s.gemini_model, s.gemini_api_key)

    problems = check_compatibility(
        index.manifest,
        model_name=s.retriever_model,
        model_revision=getattr(encoder, "model_revision", None),
        query_format_id=getattr(encoder, "query_format", CPE_0_3_18).format_id,
        index_dir=s.index_dir,
    )
    if problems:
        msg = "indeks/serve uyumsuzluğu: " + "; ".join(problems)
        if not s.allow_index_mismatch:
            raise IndexCompatibilityError(msg)
        logger.warning("BG_ALLOW_INDEX_MISMATCH=true ile devam ediliyor — %s", msg)

    def load_image(image_path: str) -> bytes:
        return (s.data_dir / image_path).read_bytes()

    if s.retrieval_pipeline == "exhaustive":
        retriever = ExhaustiveBinaryRetriever(index, meta, encoder)
    else:
        retriever = TwoStageRetriever(index, meta, encoder)

    manifest = index.manifest
    if manifest is not None:
        index_revision = (
            f"{manifest.corpus_checksum[:12]}/{manifest.query_format.format_id}/"
            f"{manifest.quantization}"
        )
        query_format_id = manifest.query_format.format_id
        quantization = manifest.quantization
    else:
        index_revision = None
        query_format_id = None
        quantization = None
    service = AskService(retriever, answerer, s.min_score_threshold, load_image)

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
    prom.set_app_info(
        pages=len(index.page_ids),
        retriever_model=s.retriever_model,
        gemini_model=s.gemini_model,
        device=s.device,
        version=app_version,
        threshold=s.min_score_threshold,
        index_revision=index_revision or "unknown",
        query_format=query_format_id or "unknown",
    )

    app = FastAPI(title="Belge-Gözü")

    def build_event(
        *,
        endpoint: str,
        status: str,
        http_status: int,
        total_ms: float,
        col: StageCollector,
        query: str,
        hits: list[PageHit],
        answer: Answer | None = None,
        error_type: str | None = None,
        k: int | None = None,
        candidates: int | None = None,
    ) -> RequestEvent:
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
            cost = (tokens_in / 1e6) * s.gemini_price_in_usd_per_1m + (
                tokens_out / 1e6
            ) * s.gemini_price_out_usd_per_1m
        honest_miss = None
        if answer is not None and not answer.abstained:
            honest_miss = "bulamadım" in answer.text.lower()  # sezgisel (spec §5)
        return RequestEvent(
            ts=datetime.now(UTC).isoformat(),
            endpoint=endpoint,
            status=status,
            http_status=http_status,
            total_ms=total_ms,
            encode_ms=col.stages.get("query_encode"),
            stage1_ms=col.stages.get("stage1_hamming"),
            stage2_ms=col.stages.get("stage2_maxsim"),
            answer_ms=answer_ms,
            top_score=top,
            margin_1_2=margin,
            abstained=answer.abstained if answer else None,
            honest_miss=honest_miss,
            k=k,
            candidates=candidates,
            query_len=len(query),
            query_text=query if s.log_query_text else None,
            query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            answer_len=len(answer.text) if answer else None,
            citations_n=len(answer.citations) if answer else None,
            tokens_in=tokens_in if isinstance(tokens_in, int) else None,
            tokens_out=tokens_out if isinstance(tokens_out, int) else None,
            tokens_per_s=tps,
            est_cost_usd=cost,
            error_type=error_type,
            pipeline=s.retrieval_pipeline,
            index_revision=index_revision,
            detail={
                "hits": [{"page_id": h.page_id, "score": h.score} for h in hits],
                "threshold": s.min_score_threshold,
                "retriever_model": s.retriever_model,
                "gemini_model": s.gemini_model,
                "device": s.device,
                "app_version": app_version,
                "stages": dict(col.stages),
                "retrieval": {"query_format": query_format_id, "quantization": quantization},
            },
        )

    def record_event(**kwargs) -> None:
        # Telemetri best-effort'tur: olay kurma/kaydetme hiçbir koşulda başarılı
        # bir isteği düşürmez.
        try:
            ev = build_event(**kwargs)
            rec.record(ev)
            prom.observe(ev)
        except Exception:
            logger.exception("telemetri olayı işlenemedi (istek etkilenmedi)")

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "pages": len(index.page_ids),
            "threshold": s.min_score_threshold,
        }

    @app.post("/search")
    def search(body: SearchBody) -> dict[str, list[PageHit]]:
        t0 = time.perf_counter()
        cand = s.stage1_candidates if s.retrieval_pipeline == "two-stage" else None
        with collecting() as col, prom.inflight("/search"):
            try:
                k = body.k or s.top_k
                if cand is None:
                    hits = retriever.search(body.query, k=k)
                else:
                    # cand is not None <=> retrieval_pipeline == "two-stage" (TwoStageRetriever);
                    # pyright can't narrow retriever's union type from cand's nullability.
                    hits = retriever.search(body.query, k=k, candidates=cand)  # pyright: ignore[reportCallIssue]
            except Exception as e:
                record_event(
                    endpoint="/search",
                    status="error",
                    http_status=500,
                    total_ms=(time.perf_counter() - t0) * 1000,
                    col=col,
                    query=body.query,
                    hits=[],
                    error_type=type(e).__name__,
                )
                raise
            record_event(
                endpoint="/search",
                status="ok",
                http_status=200,
                total_ms=(time.perf_counter() - t0) * 1000,
                col=col,
                query=body.query,
                hits=hits,
                k=k,
                candidates=cand,
            )
        return {"hits": hits}

    @app.post("/ask")
    def ask(body: AskBody) -> dict:
        t0 = time.perf_counter()
        cand = s.stage1_candidates if s.retrieval_pipeline == "two-stage" else None
        with collecting() as col, prom.inflight("/ask"):
            try:
                answer, hits = service.ask(body.question, k=s.top_k, candidates=cand)
            except Exception as e:
                record_event(
                    endpoint="/ask",
                    status="error",
                    http_status=500,
                    total_ms=(time.perf_counter() - t0) * 1000,
                    col=col,
                    query=body.question,
                    hits=[],
                    error_type=type(e).__name__,
                )
                raise
            if col.notes.get("degraded"):
                status = "degraded"
            elif answer.abstained:
                status = "abstained"
            else:
                status = "answered"
            record_event(
                endpoint="/ask",
                status=status,
                http_status=200,
                total_ms=(time.perf_counter() - t0) * 1000,
                col=col,
                query=body.question,
                hits=hits,
                answer=answer,
                k=s.top_k,
                candidates=cand,
            )
        return {"answer": answer.model_dump(), "hits": [h.model_dump() for h in hits]}

    @app.get("/metrics")
    def metrics() -> Response:
        body, ctype = prom.render()
        return Response(content=body, media_type=ctype)

    @app.get("/stats")
    def stats() -> dict:
        # /stats bir telemetri okuma uç noktasıdır: herhangi bir sqlite hatası
        # isteği asla 500'e düşürmez, sıfırlanmış bir gövdeye geriler.
        degraded = {
            "requests": 0,
            "avg_ms": 0.0,
            "p95_ms": 0.0,
            "abstain_rate": 0.0,
            "by_endpoint": {},
        }
        db = None
        try:
            db = sqlite3.connect(rec.db_path)
            n, avg = db.execute("SELECT COUNT(*), COALESCE(AVG(total_ms),0) FROM events").fetchone()
            vals = [
                r[0] for r in db.execute("SELECT total_ms FROM events ORDER BY id DESC LIMIT 10000")
            ]
            vals.sort()
            p95 = vals[min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)] if vals else 0.0
            ab = db.execute(
                "SELECT COALESCE(AVG(abstained),0) FROM events "
                "WHERE endpoint='/ask' AND status <> 'degraded'"
            ).fetchone()[0]
            by = dict(db.execute("SELECT endpoint, COUNT(*) FROM events GROUP BY endpoint"))
            return {
                "requests": n,
                "avg_ms": round(avg, 1),
                "p95_ms": round(p95, 1),
                "abstain_rate": round(ab, 3),
                "by_endpoint": by,
            }
        except Exception:
            logger.exception("/stats okunamadı, sıfırlanmış gövdeye gerilendi")
            return degraded
        finally:
            if db is not None:
                db.close()

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
