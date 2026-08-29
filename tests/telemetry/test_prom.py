from fastapi.testclient import TestClient

from belge_gozu.answer.base import Answer
from belge_gozu.app.main import create_app
from belge_gozu.config import Settings
from belge_gozu.telemetry.prom import PromMetrics
from belge_gozu.telemetry.schema import RequestEvent


class _StubAnswerer:
    def answer(self, question, pages, image_loader):
        return Answer(text=f"yanıt: {question}", citations=[pages[0].page_id])


def _ask_ev(**kw) -> RequestEvent:
    base = dict(
        ts="t",
        endpoint="/ask",
        status="answered",
        http_status=200,
        total_ms=9000.0,
        encode_ms=1500.0,
        stage1_ms=8.0,
        stage2_ms=40.0,
        answer_ms=7000.0,
        top_score=60.9,
        margin_1_2=0.2,
        abstained=False,
        tokens_in=5000,
        tokens_out=210,
        tokens_per_s=30.0,
        est_cost_usd=0.0006,
        query_sha256="d" * 64,
    )
    base.update(kw)
    return RequestEvent(**base)


def test_observe_and_render_contains_series():
    pm = PromMetrics()
    pm.set_app_info(
        pages=4222,
        retriever_model="colsmol",
        gemini_model="gf",
        device="cpu",
        version="0.1.0",
        threshold=60.0,
        index_revision="abc123456789/cpe-0.3.18/sign-1bit",
        query_format="cpe-0.3.18",
    )
    pm.observe(_ask_ev())
    pm.observe(
        _ask_ev(
            status="abstained",
            abstained=True,
            answer_ms=None,
            tokens_in=None,
            tokens_out=None,
            tokens_per_s=None,
            est_cost_usd=None,
        )
    )
    body, ctype = pm.render()
    text = body.decode()
    assert 'bg_http_requests_total{endpoint="/ask",status="answered"} 1.0' in text
    assert 'bg_abstain_total{reason="threshold"} 1.0' in text
    assert "bg_request_duration_seconds_bucket" in text
    assert 'bg_stage_duration_seconds_bucket{le="2.0",stage="query_encode"}' in text
    assert 'bg_llm_tokens_total{direction="output"} 210.0' in text
    assert "bg_index_pages 4222.0" in text
    assert "openmetrics" in ctype or "text/plain" in ctype


def test_score_histograms_carry_quantization_label():
    """Skor/marj örnekleri TEMSİLE göre etiketlenir (T14).

    Skorun ölçeği kuantizasyona bağlıdır (binary 0-128 vs normalize
    [-1,1]); etiketsiz tek seride geçiş öncesi/sonrası örnekler geri
    dönülemez biçimde karışır ve histogram quantile'ları anlamsızlaşır.
    Değer olayın kendi künyesinden (`detail.retrieval.quantization`) gelir."""
    pm = PromMetrics()
    pm.observe(
        _ask_ev(
            top_score=0.62,
            margin_1_2=0.01,
            detail={"retrieval": {"query_format": "train-compat-v1", "quantization": "int8"}},
        )
    )
    text = pm.render()[0].decode()
    assert 'bg_retrieval_top_score_bucket{le="0.65",quantization="int8"} 1.0' in text
    assert 'bg_retrieval_score_margin_bucket{le="0.02",quantization="int8"} 1.0' in text
    # eşik civarındaki bucket'lar gerçekten normalize ölçekte
    assert 'bg_retrieval_top_score_bucket{le="0.58",quantization="int8"} 0.0' in text


def test_bm25_routing_set_is_derived_from_the_single_scale_map():
    """Yönlendirme kümesi `config.PIPELINE_SCORE_SCALE`'den TÜRETİLMELİ (review M1).

    Elle yazılmış bir kopya, ölçeği BM25 olan yeni bir pipeline eklendiğinde
    sessizce eskir: korkuluk ve uyarı doğru davranır ama skorlar normalize
    [-1,1] histogramına dökülmeye başlar — T14 hatasının aynısı."""
    from belge_gozu.config import BM25_SCALE, PIPELINE_SCORE_SCALE
    from belge_gozu.telemetry.prom import BM25_SCALE_PIPELINES

    assert BM25_SCALE_PIPELINES == {p for p, s in PIPELINE_SCORE_SCALE.items() if s == BM25_SCALE}
    assert BM25_SCALE_PIPELINES == {"hybrid"}  # bugünkü durum
    # ölçek adının kendisi de tek kaynaktan gelmeli (dize kopyası yok)
    assert PIPELINE_SCORE_SCALE["hybrid"] == BM25_SCALE


def test_hybrid_scores_go_to_the_bm25_series():
    """Skor ÖLÇEĞİ pipeline'a bağlı (P1): hibrit örnekler AYRI seriye düşer.

    Aynı histogramda toplanırlarsa geri dönülemez biçimde karışırlar — BM25
    (üst sınırsız, ~4-70) normalize [-1,1] bucket'larının hepsini aşar ve
    quantile'ları anlamsızlaştırır."""
    pm = PromMetrics()
    pm.observe(
        _ask_ev(
            pipeline="hybrid",
            top_score=26.05,
            margin_1_2=3.4,
            detail={"retrieval": {"query_format": "train-compat-v1", "quantization": "int8"}},
        )
    )
    text = pm.render()[0].decode()
    assert 'bg_retrieval_top_score_bm25_bucket{le="30.0"} 1.0' in text
    assert 'bg_retrieval_top_score_bm25_bucket{le="20.0"} 0.0' in text
    # eşik 10.6 bucket sınırı olarak var (çalışma noktası doğrudan okunabilsin)
    assert 'bg_retrieval_top_score_bm25_bucket{le="10.6"} 0.0' in text
    assert 'bg_retrieval_score_margin_bm25_bucket{le="5.0"} 1.0' in text
    # görsel-ölçek serisine HİÇ örnek düşmemeli (etiketli seri hiç doğmaz)
    assert "bg_retrieval_top_score_bucket" not in text
    assert "bg_retrieval_score_margin_bucket" not in text


def test_visual_pipelines_keep_using_the_normalized_series():
    """Karşı taraf: görsel kollar eski seride kalır (etiketiyle birlikte)."""
    pm = PromMetrics()
    pm.observe(
        _ask_ev(
            pipeline="exhaustive",
            top_score=0.62,
            margin_1_2=0.01,
            detail={"retrieval": {"query_format": "train-compat-v1", "quantization": "int8"}},
        )
    )
    text = pm.render()[0].decode()
    assert 'bg_retrieval_top_score_bucket{le="0.65",quantization="int8"} 1.0' in text
    # BM25 serisi (etiketsiz) hep render edilir; ÖRNEK ALMAMIŞ olmalı
    assert "bg_retrieval_top_score_bm25_count 0.0" in text
    assert "bg_retrieval_score_margin_bm25_count 0.0" in text


def test_score_histogram_without_identity_falls_back_to_unknown():
    """Künye taşımayan olay (eski satır / enjekte edilmiş olay) düşürülmez."""
    pm = PromMetrics()
    pm.observe(_ask_ev(top_score=0.5, margin_1_2=0.0, detail={}))
    assert 'bg_retrieval_top_score_bucket{le="0.5",quantization="unknown"} 1.0' in (
        pm.render()[0].decode()
    )


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


def test_observe_records_uncovered_stage_names():
    """_STAGE_COLS dışındaki aşamalar (ör. exhaustive_maxsim) da histogram'a düşmeli."""
    pm = PromMetrics()
    pm.observe(_ask_ev(detail={"stages": {"exhaustive_maxsim": 12.0, "query_encode": 5.0}}))
    text = pm.render()[0].decode()
    assert 'bg_stage_duration_seconds_bucket{le="+Inf",stage="exhaustive_maxsim"} 1.0' in text


def test_metrics_endpoint_exposes_exhaustive_stage_and_index_revision(tiny_corpus):
    data_dir, enc, _ = tiny_corpus
    settings = Settings(
        data_dir=data_dir,
        index_dir=data_dir / "index",
        retrieval_pipeline="exhaustive",
        min_score_threshold=-1e9,
    )
    app = create_app(settings=settings, encoder=enc, answerer=_StubAnswerer())
    c = TestClient(app)
    c.post("/search", json={"query": "deneme sorgusu"})
    r = c.get("/metrics")
    text = r.text
    assert r.status_code == 200
    assert 'bg_stage_duration_seconds_bucket{le="+Inf",stage="exhaustive_maxsim"}' in text
    assert "index_revision=" in text


def test_metrics_endpoint_exposes_hybrid_stages_end_to_end(tiny_corpus):
    """Uçtan uca: yeni aşama adları detail.stages fallback'inden Prometheus'a
    kendiliğinden akıyor (prom.py'de aşama adı listesi TUTULMUYOR) ve skor
    BM25 serisine düşüyor."""
    data_dir, enc, _ = tiny_corpus
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index", min_score_threshold=-1e9)
    c = TestClient(create_app(settings=settings, encoder=enc, answerer=_StubAnswerer()))
    c.post("/search", json={"query": "yerleşim yeri nedir"})
    text = c.get("/metrics").text
    assert 'bg_stage_duration_seconds_bucket{le="+Inf",stage="text_bm25"}' in text
    assert 'bg_stage_duration_seconds_bucket{le="+Inf",stage="route_fuse"}' in text
    assert 'bg_stage_duration_seconds_bucket{le="+Inf",stage="exhaustive_maxsim"}' in text
    assert "bg_retrieval_top_score_bm25_bucket" in text
