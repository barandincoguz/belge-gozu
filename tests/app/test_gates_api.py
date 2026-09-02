"""P2 iki kapının SERVİS yüzeyi: `/ask` gövdesi, olay künyesi, `/metrics`.

`tests/app/test_api.py` BU COMMIT'TE HİÇ DEĞİŞMEDİ — bayrak-kapalı
değişmezliğin asıl kanıtı odur. Buradaki testler o kilidin üstüne iki şey
ekler: (a) bayrak kapalıyken gövde/olay künyesinde kapı anahtarlarının
BULUNMADIĞI, (b) bayrak açıkken kapıların gerçekten koştuğu.
"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from belge_gozu.answer.base import VERIFIER_DEMOTE_TEXT, Answer
from belge_gozu.answer.calibrate import (
    FEATURE_ORDER,
    CalibrationArtifact,
    Calibrator,
    calibration_key,
)
from belge_gozu.app.main import create_app
from belge_gozu.config import Settings
from belge_gozu.index.compat import IndexCompatibilityError
from belge_gozu.index.manifest import read_manifest


class CitingAnswerer:
    """Atıflı, tek iddialık bir yanıt — kanıt kapısının çalışabildiği en sade girdi."""

    def answer(self, question, pages, image_loader):
        return Answer(
            text="Yerleşim yeri sürekli kalma niyetiyle oturulan yerdir [S1].",
            citations=[pages[0].page_id],
        )


class StubVerifierClient:
    def __init__(self, verdict: str = "supported"):
        self.verdict = verdict
        self.prompts: list[str] = []

    def generate_json(self, prompt, schema=None):
        self.prompts.append(prompt)
        return json.dumps({"verdict": self.verdict, "gerekce": "stub"})


def _revision(data_dir) -> str:
    m = read_manifest(data_dir / "index")
    assert m is not None
    return f"{m.corpus_checksum[:12]}/{m.query_format.format_id}/{m.quantization}"


def _write_calibrator(data_dir, tau: float) -> str:
    """Ağırlıkları SIFIR bir kalibratör: p her sorguda 0.5, tau testin kolu."""
    revision = _revision(data_dir)
    key = calibration_key(revision, "hybrid")
    d = len(FEATURE_ORDER)
    CalibrationArtifact(
        key=key,
        index_revision=revision,
        pipeline="hybrid",
        recipe_fingerprint=key.split("__")[-1],
        calibrator=Calibrator(
            feature_names=FEATURE_ORDER,
            mean=(0.0,) * d,
            std=(1.0,) * d,
            weights=(0.0,) * d,
            bias=0.0,
            fit_info={},
        ),
        thresholds={"chosen": {"name": "t", "value": tau, "statistical_guarantee": "none"}},
        kunye={},
    ).save(data_dir / "calibration" / key)
    return key


def _event_detail(data_dir) -> dict:
    db = sqlite3.connect(data_dir / "requests.sqlite")
    try:
        (raw,) = db.execute("SELECT detail FROM events WHERE endpoint='/ask'").fetchone()
    finally:
        db.close()
    return json.loads(raw)


def _client(tiny_corpus, verdict="supported", tau=0.4, monkeypatch=None, **flags) -> TestClient:
    data_dir, enc, _ = tiny_corpus
    if flags.get("gate_calibrated"):
        _write_calibrator(data_dir, tau)
    if flags.get("gate_verifier"):
        stub = StubVerifierClient(verdict)
        assert monkeypatch is not None
        monkeypatch.setattr("belge_gozu.answer.verify.GeminiVerifierClient", lambda *a, **kw: stub)
    settings = Settings(
        data_dir=data_dir, index_dir=data_dir / "index", min_score_threshold=-1e9, **flags
    )
    return TestClient(create_app(settings=settings, encoder=enc, answerer=CitingAnswerer()))


# --- bayrak KAPALI: gövde ve olay künyesi P1 ile birebir ----------------------


def test_flags_off_body_has_no_detail_key_and_event_has_no_gate_blocks(tiny_corpus):
    data_dir, _, _ = tiny_corpus
    body = _client(tiny_corpus).post("/ask", json={"question": "yerleşim yeri nedir"}).json()
    assert set(body) == {"status", "honest_miss", "no_match", "answer", "hits"}
    assert body["status"] == "answered"
    detail = _event_detail(data_dir)
    assert "gate1" not in detail and "gate2" not in detail


def test_flags_off_metrics_has_no_verifier_series_samples(tiny_corpus):
    c = _client(tiny_corpus)
    c.post("/ask", json={"question": "yerleşim yeri nedir"})
    text = c.get("/metrics").text
    # Seri KATALOGDA ve registry'de var (HELP satırı) ama ÖRNEĞİ yok.
    assert "# HELP bg_verifier_verdicts_total" in text
    assert "bg_verifier_verdicts_total{verdict=" not in text


# --- kapı 1: kalibre getirim --------------------------------------------------


def test_gate1_on_records_p_and_tau_in_body_and_event(tiny_corpus):
    data_dir, _, _ = tiny_corpus
    c = _client(tiny_corpus, tau=0.4, gate_calibrated=True)
    body = c.post("/ask", json={"question": "yerleşim yeri nedir"}).json()
    assert body["status"] == "answered"
    g1 = body["detail"]["gate1"]
    assert g1["p"] == pytest.approx(0.5) and g1["tau"] == 0.4 and g1["passed"] is True
    assert set(g1["features"]) == set(FEATURE_ORDER)
    assert _event_detail(data_dir)["gate1"]["p"] == pytest.approx(0.5)


def test_gate1_abstains_below_tau(tiny_corpus):
    c = _client(tiny_corpus, tau=0.9, gate_calibrated=True)
    body = c.post("/ask", json={"question": "yerleşim yeri nedir"}).json()
    assert body["status"] == "abstained" and body["detail"]["gate1"]["passed"] is False


def test_missing_calibration_artifact_stops_startup(tiny_corpus):
    data_dir, enc, _ = tiny_corpus
    settings = Settings(
        data_dir=data_dir,
        index_dir=data_dir / "index",
        min_score_threshold=-1e9,
        gate_calibrated=True,
    )
    with pytest.raises(IndexCompatibilityError, match="calibrate fit"):
        create_app(settings=settings, encoder=enc, answerer=CitingAnswerer())


# --- kapı 2: kanıt ------------------------------------------------------------


def test_gate2_supported_answer_stays_answered_and_counts_a_verdict(tiny_corpus, monkeypatch):
    c = _client(tiny_corpus, verdict="supported", monkeypatch=monkeypatch, gate_verifier=True)
    body = c.post("/ask", json={"question": "yerleşim yeri nedir"}).json()
    assert body["status"] == "answered" and body["answer"]["citations"]
    assert body["detail"]["gate2"]["demoted"] is False
    assert 'bg_verifier_verdicts_total{verdict="supported"} 1.0' in c.get("/metrics").text


def test_gate2_demote_keeps_the_status_vocabulary_and_flags_it_in_detail(tiny_corpus, monkeypatch):
    data_dir, _, _ = tiny_corpus
    c = _client(tiny_corpus, verdict="unsupported", monkeypatch=monkeypatch, gate_verifier=True)
    body = c.post("/ask", json={"question": "yerleşim yeri nedir"}).json()
    # status SÖZLÜĞÜ genişlemedi: düşürme de "abstained"tır (arayüz kilidi).
    assert body["status"] == "abstained"
    assert body["answer"]["text"] == VERIFIER_DEMOTE_TEXT
    assert body["answer"]["citations"] == []
    assert body["honest_miss"] is False
    assert body["detail"]["gate2"]["demoted"] is True
    assert body["hits"], "düşürülen yanıtta bile sayfalar gösterilir"
    assert _event_detail(data_dir)["gate2"]["claims"][0]["verdict"] == "unsupported"
    assert 'bg_verifier_verdicts_total{verdict="unsupported"} 1.0' in c.get("/metrics").text


def test_gate2_verifier_reads_the_served_page_text(tiny_corpus, monkeypatch):
    stub = StubVerifierClient("supported")
    monkeypatch.setattr("belge_gozu.answer.verify.GeminiVerifierClient", lambda *a, **kw: stub)
    data_dir, enc, _ = tiny_corpus
    settings = Settings(
        data_dir=data_dir,
        index_dir=data_dir / "index",
        min_score_threshold=-1e9,
        gate_verifier=True,
    )
    c = TestClient(create_app(settings=settings, encoder=enc, answerer=CitingAnswerer()))
    c.post("/ask", json={"question": "yerleşim yeri nedir"})
    # Kanıt, BM25'in skorladığı `page_texts.parquet` metninin ta kendisi.
    assert "TÜRK MEDENİ KANUNU" in stub.prompts[0]


def test_both_gates_on_together(tiny_corpus, monkeypatch):
    c = _client(
        tiny_corpus,
        verdict="supported",
        tau=0.4,
        monkeypatch=monkeypatch,
        gate_calibrated=True,
        gate_verifier=True,
    )
    body = c.post("/ask", json={"question": "yerleşim yeri nedir"}).json()
    assert body["status"] == "answered"
    assert set(body["detail"]) == {"gate1", "gate2"}


# --- fix round 1: bütçe kablolaması, sebep ekseni, erken fail-fast ------------


def test_serve_wires_a_per_request_budget_that_actually_caps_attempts(tiny_corpus, monkeypatch):
    """review H1: `serve` yolunda bütçe VARDI ama HİÇ BAĞLANMAMIŞTI.

    `verifier_max_llm_calls=1` ile iki iddialı bir yanıt: ikinci iddia çağrı
    YAPILMADAN `belirsiz` olur, yanıt düşer ve olay bunu söyler."""
    stub = StubVerifierClient("supported")
    monkeypatch.setattr("belge_gozu.answer.verify.GeminiVerifierClient", lambda *a, **kw: stub)
    data_dir, enc, _ = tiny_corpus

    class TwoClaimAnswerer:
        def answer(self, question, pages, image_loader):
            return Answer(
                text=(
                    "Yerleşim yeri sürekli kalma niyetiyle oturulan yerdir [S1]. "
                    "Bir kimsenin birden çok yerleşim yeri olamaz [S1]."
                ),
                citations=[pages[0].page_id],
            )

    settings = Settings(
        data_dir=data_dir,
        index_dir=data_dir / "index",
        min_score_threshold=-1e9,
        gate_verifier=True,
        verifier_max_llm_calls=1,
    )
    c = TestClient(create_app(settings=settings, encoder=enc, answerer=TwoClaimAnswerer()))
    body = c.post("/ask", json={"question": "yerleşim yeri nedir"}).json()
    g2 = body["detail"]["gate2"]
    assert g2["budget_max_attempts"] == 1 and g2["budget_used"] == 1
    assert g2["budget_exhausted"] is True and g2["api_attempts"] == 1
    assert len(stub.prompts) == 1, "tavan GERÇEKTEN ikinci çağrıyı kesti"
    assert body["status"] == "abstained" and g2["demoted"] is True
    assert [cl["verdict"] for cl in g2["claims"]] == ["supported", "belirsiz"]
    # Bütçe İSTEK başınadır: ikinci istek taze tavanla gelir.
    c.post("/ask", json={"question": "yerleşim yeri nedir"})
    assert len(stub.prompts) == 2


def test_abstain_reason_distinguishes_threshold_gate1_and_gate2(tiny_corpus, monkeypatch):
    """review M2: üç ayrı fren üç ayrı etiket."""
    data_dir, enc, _ = tiny_corpus

    # (a) eski eşik
    high = Settings(data_dir=data_dir, index_dir=data_dir / "index", min_score_threshold=100.0)
    c_a = TestClient(create_app(settings=high, encoder=enc, answerer=CitingAnswerer()))
    c_a.post("/ask", json={"question": "yerleşim yeri nedir"})
    assert 'bg_abstain_total{reason="threshold"} 1.0' in c_a.get("/metrics").text

    # (b) kapı 1
    c_b = _client(tiny_corpus, tau=0.9, gate_calibrated=True)
    c_b.post("/ask", json={"question": "yerleşim yeri nedir"})
    assert 'bg_abstain_total{reason="gate1"} 1.0' in c_b.get("/metrics").text

    # (c) kapı 2 düşürmesi
    c_c = _client(tiny_corpus, verdict="unsupported", monkeypatch=monkeypatch, gate_verifier=True)
    c_c.post("/ask", json={"question": "yerleşim yeri nedir"})
    metrics = c_c.get("/metrics").text
    assert 'bg_abstain_total{reason="gate2_demote"} 1.0' in metrics
    assert 'bg_abstain_total{reason="threshold"}' not in metrics


def test_calibrator_failfast_happens_before_the_heavy_encoder_load(tiny_corpus):
    """review L1: tek satırlık `calibrate fit` mesajı için VLM yüklenmemeli."""
    data_dir, _, _ = tiny_corpus

    class ExplodingEncoder:
        """Kullanılırsa test anlamını kaybeder: kontrol ONDAN ÖNCE olmalı."""

        def encode_pages(self, images):
            raise AssertionError("ağır yükleme yapılmamalıydı")

        def encode_query(self, text):
            raise AssertionError("ağır yükleme yapılmamalıydı")

    settings = Settings(
        data_dir=data_dir,
        index_dir=data_dir / "index",
        min_score_threshold=-1e9,
        gate_calibrated=True,
    )
    with pytest.raises(IndexCompatibilityError, match="calibrate fit"):
        create_app(settings=settings, encoder=ExplodingEncoder(), answerer=CitingAnswerer())
