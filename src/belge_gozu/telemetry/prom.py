from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from belge_gozu.config import BM25_SCALE, pipelines_on_scale
from belge_gozu.telemetry.schema import RequestEvent

REQUEST_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30)
STAGE_BUCKETS = (0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20)
# T14: skor/marj bucket'ları normalize [-1,1] ölçeğine taşındı (eşik 0.58
# çevresinde sıklaştırılmış). GEÇİŞ ÖNCESİ seriler/satırlar eski binary
# ölçeğindedir (0-128) ve aynı seride karışırlar — hangi ölçekte oldukları
# `bg_app_info`'nun `index_revision`/`threshold` etiketlerinden (olay
# tablosunda `index_revision` kolonundan) ve bu histogramların
# `quantization` etiketinden ayırt edilir.
SCORE_BUCKETS = (0.30, 0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70, 0.80)
MARGIN_BUCKETS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)
# P1: hibrit yolda skoru BM25 üretir — kalibre edilmemiş, ÜST SINIRSIZ birim
# (ölçülen top-1 bandı ~4-70). Normalize [-1,1] bucket'larıyla aynı seride
# toplanamaz (hepsi son bucket'a düşer, quantile'lar anlamsızlaşır), bu yüzden
# AYRI seri: eşik 10.6 çevresinde sıklaştırılmış.
BM25_SCORE_BUCKETS = (0, 5, 10, 10.6, 15, 20, 30, 45, 70, 100)
BM25_MARGIN_BUCKETS = (0, 0.5, 1, 2, 5, 10, 20, 40)
TPS_BUCKETS = (5, 10, 20, 40, 80, 160)

# Skorun ölçeğini pipeline belirler; yönlendirme bunu `config`ten TÜRETİR,
# kopya sabit TUTMAZ (review M1). Elle yazılmış bir kopya, ölçeği BM25 olan
# yeni bir pipeline eklendiğinde sessizce eskir ve o kolun skorları normalize
# [-1,1] histogramına dökülür — commit'in önlemek için yazıldığı T14 hatasının
# aynısı. Olay künyesi zaten `pipeline` taşıyor.
BM25_SCALE_PIPELINES = pipelines_on_scale(BM25_SCALE)

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
        self.requests = Counter(
            "bg_http_requests", "İstek sayısı", ["endpoint", "status"], registry=r
        )
        self.duration = Histogram(
            "bg_request_duration_seconds",
            "Uçtan uca süre",
            ["endpoint"],
            buckets=REQUEST_BUCKETS,
            registry=r,
        )
        self.stage = Histogram(
            "bg_stage_duration_seconds",
            "Aşama süresi",
            ["stage"],
            buckets=STAGE_BUCKETS,
            registry=r,
        )
        # `quantization` etiketi (T14): skorun ölçeği hangi temsille
        # üretildiğine bağlıdır (binary 0-128 vs normalize [-1,1]). Etiketsiz
        # tek seride geçiş öncesi ve sonrası örnekler GERİ DÖNÜLEMEZ biçimde
        # karışırdı — histogram toplamları ve quantile'lar anlamsızlaşırdı.
        self.top_score = Histogram(
            "bg_retrieval_top_score",
            "En iyi skor",
            ["quantization"],
            buckets=SCORE_BUCKETS,
            registry=r,
        )
        self.margin = Histogram(
            "bg_retrieval_score_margin",
            "top1-top2 farkı",
            ["quantization"],
            buckets=MARGIN_BUCKETS,
            registry=r,
        )
        # P1 hibrit kolun BM25-ölçekli karşılıkları. `quantization` etiketi
        # YOK: BM25 skoru indeks temsilinden bağımsızdır (metin katmanından
        # gelir), etiket burada yanıltıcı olurdu.
        self.top_score_bm25 = Histogram(
            "bg_retrieval_top_score_bm25",
            "En iyi skor (BM25 ölçeği, hibrit pipeline)",
            buckets=BM25_SCORE_BUCKETS,
            registry=r,
        )
        self.margin_bm25 = Histogram(
            "bg_retrieval_score_margin_bm25",
            "top1-top2 farkı (BM25 ölçeği, hibrit pipeline)",
            buckets=BM25_MARGIN_BUCKETS,
            registry=r,
        )
        self.abstain = Counter("bg_abstain", "Abstain sayısı", ["reason"], registry=r)
        # review L3 (2026-08-30): 429'lar `collecting()`/`record_event`den ÖNCE
        # fırlatıldığı için hiçbir RequestEvent'e girmez (getirici çağrılmaz,
        # bu bilinçli) — bu sayaç olmadan sınırlayıcının çalıştığına dair
        # `/metrics`'te hiçbir iz kalmazdı. `app/main.py::enforce_rate_limit`
        # doğrudan artırır (`.observe(ev)` yolunu KULLANMAZ).
        self.rate_limited = Counter(
            "bg_rate_limited", "429 hız sınırı reddi", ["endpoint"], registry=r
        )
        # Y23: reddedilen isteklerin BİRLEŞİK sayacı — 422 (içeriksiz sorgu) ve
        # 429 (hız sınırı) tek bir `reason` ekseninde. `bg_rate_limited_total`
        # KALDIRILMADI, çünkü o serinin ekseni `endpoint`tir ve farklı bir
        # soruya cevap verir ("hangi uç nokta sınırlanıyor?"); bu seri
        # "istekler NEDEN reddediliyor?" sorusunu cevaplar. Aynı 429 iki seride
        # görünür ama TEK seri içinde iki kez sayılmaz.
        #
        # 422'nin `bg_rate_limited_total`a hiç girmemesi bilinçliydi (framework
        # düzeyi ret); bu sayaç o kör noktayı kapatır: "sistem kaç kere
        # içeriksiz sorgu reddetti?" sorusunun cevabı P2 için tam olarak
        # "cevaplanmaması gereken"in en temiz sınıfıdır.
        self.rejected = Counter("bg_rejected", "Reddedilen istek (422/429)", ["reason"], registry=r)
        self.honest_miss = Counter("bg_honest_miss", "'bulamadım' yanıtları", registry=r)
        self.tokens = Counter("bg_llm_tokens", "LLM token sayısı", ["direction"], registry=r)
        self.tps = Histogram(
            "bg_llm_tokens_per_second", "Üretim hızı", buckets=TPS_BUCKETS, registry=r
        )
        self.cost = Counter("bg_llm_cost_usd", "Tahmini maliyet (USD)", registry=r)
        self.inflight_g = Gauge("bg_inflight_requests", "Anlık istek", ["endpoint"], registry=r)
        self.pages = Gauge("bg_index_pages", "Dizindeki sayfa sayısı", registry=r)
        self.info = Info("bg_app", "Uygulama künyesi", registry=r)

    def set_app_info(
        self,
        *,
        pages: int,
        retriever_model: str,
        gemini_model: str,
        device: str,
        version: str,
        threshold: float,
        index_revision: str,
        query_format: str,
    ) -> None:
        self.pages.set(pages)
        self.info.info(
            {
                "retriever_model": retriever_model,
                "gemini_model": gemini_model,
                "device": device,
                "version": version,
                "threshold": str(threshold),
                "index_revision": index_revision,
                "query_format": query_format,
            }
        )

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
        # _STAGE_COLS'ta kolonu olmayan aşamalar (ör. exhaustive_maxsim) detail'den gelir.
        for stage_name, ms in ev.detail.get("stages", {}).items():
            if stage_name not in _STAGE_COLS and ms is not None:
                self.stage.labels(stage=stage_name).observe(ms / 1000.0)
        # Skor ÖLÇEĞİ pipeline'a bağlı (P1): hibrit BM25 birimi, görsel kollar
        # normalize [-1,1]. İkisi aynı seride toplanırsa geri dönülemez biçimde
        # karışır (T14'te aynı hata kuantizasyon ekseninde yaşandı), bu yüzden
        # olayın kendi `pipeline` künyesine göre AYRI serilere yönlendirilir.
        if ev.pipeline in BM25_SCALE_PIPELINES:
            if ev.top_score is not None:
                self.top_score_bm25.observe(ev.top_score)
            if ev.margin_1_2 is not None:
                self.margin_bm25.observe(ev.margin_1_2)
        else:
            # Görsel ölçek: değer TEMSİLE göre etiketlenir (app/main.py
            # `detail.retrieval`'i manifest'ten doldurur); künye taşımayan
            # olaylar "unknown"a düşer.
            quant = str((ev.detail.get("retrieval") or {}).get("quantization") or "unknown")
            if ev.top_score is not None:
                self.top_score.labels(quantization=quant).observe(ev.top_score)
            if ev.margin_1_2 is not None:
                self.margin.labels(quantization=quant).observe(ev.margin_1_2)
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
