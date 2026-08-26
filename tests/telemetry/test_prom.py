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
